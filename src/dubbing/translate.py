"""Phase 2: context- and gender-aware translation via LM Studio (gemma-4-31b-qat).

VRAM-managed: starts the LM Studio server, loads the LLM with a context length
computed from free VRAM, translates the transcript any -> any in sliding windows
(so style/terminology stay consistent), then unloads the model to free VRAM for
TTS. For a Czech target, output is additionally normalized for the TTS step.

This phase only talks HTTP to LM Studio, so it does not itself hold GPU memory
beyond what LM Studio uses for the LLM.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing import vram  # noqa: E402
from dubbing.config import (LLM_WEIGHTS_GB, LMSTUDIO_API,  # noqa: E402
                            LMSTUDIO_MODEL)
from dubbing.schema import DubProject  # noqa: E402
from tts_service.cz_normalize import normalize_text  # noqa: E402

# Endpoint/model — overridable via CLI for a remote LM Studio host (e.g. a Mac Studio).
API = LMSTUDIO_API
MODEL = LMSTUDIO_MODEL


def _clean_text(txt: str) -> str:
    """Strip artifacts a model may place inside the JSON string value, e.g. a
    leading 'text:' label or wrapping quotes ('text: "..."' -> '...')."""
    txt = txt.strip()
    m = re.match(r'^\s*"?text"?\s*[:=]\s*(.*)$', txt, re.IGNORECASE | re.DOTALL)
    if m:
        txt = m.group(1).strip()
    if len(txt) >= 2 and txt[0] in "\"'" and txt[-1] == txt[0]:
        txt = txt[1:-1].strip()
    return txt

WINDOW = 8             # segments per LLM call (smaller = more robust)
CONTEXT_LINES = 3      # preceding lines passed for continuity


def _is_degenerate(text: str) -> bool:
    """Detect LLM repetition loops (e.g. 'společnosti společnosti společnosti ...')."""
    words = text.split()
    if len(words) < 12:
        return False
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 5:
            return True
    from collections import Counter
    return Counter(words).most_common(1)[0][1] > max(8, int(0.35 * len(words)))


_REASONING_RE = re.compile(
    r"\b(let me|wait[, ]|re-?translat|i need to|i should|i'?ll|the sentence|"
    r"the translation|let'?s\b|actually[, ]|hmm|i think|note that|in czech|"
    r"here is|here'?s|first,|okay[, ])", re.IGNORECASE)


def _is_contaminated(src: str, tgt: str) -> bool:
    """True if a Czech-target line looks like leaked LLM reasoning / meta text."""
    if not tgt:
        return True
    if _REASONING_RE.search(tgt):
        return True
    return len(tgt) > max(80, 3 * len(src))


def _lms(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["lms", *args], capture_output=True, text=True, check=check)


def ensure_server() -> None:
    try:
        _lms("server", "start")
    except subprocess.CalledProcessError as e:
        print(f"[translate] lms server start: {e.stderr or e.stdout}")


def load_llm(context_length: int) -> None:
    print(f"[translate] loading {MODEL} (ctx={context_length}, gpu=max)")
    _lms("load", MODEL, "--gpu", "max",
         "--context-length", str(context_length), "--identifier", "dub-llm", "--yes")


def unload_llm() -> None:
    _lms("unload", "--all", check=False)


def _extract_json(text: str) -> dict:
    """Parse the JSON object from a reply, tolerant of code fences / stray prose."""
    text = re.sub(r"```(?:json)?", "", text)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in response")
    return json.loads(text[i:j + 1])


def _chat(messages: list[dict], max_tokens: int = 16384, temperature: float = 0.3,
          timeout: int = 600) -> str:
    # NB: we deliberately do NOT send response_format/json_schema. For reasoning
    # ("thinking") models LM Studio applies the schema to the thinking stream too,
    # so the model crams its reasoning into the JSON (e.g. "wait, let me retranslate").
    # Instead we let it think (reasoning -> reasoning_content) and read the final
    # JSON from content after the thinking phase.
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(API + "/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:  # rare: model put everything in the reasoning stream
        content = (msg.get("reasoning_content") or "").strip()
    return content


def _system_prompt(src: str, tgt: str) -> str:
    tgt_rules = ""
    if tgt == "cs":
        tgt_rules = (
            " The target is Czech. Be rigorous about Czech morphology: every "
            "adjective must agree with its noun in case, gender and number; use the "
            "correct case after each preposition (e.g. 'v rušném hlavním městě', NOT "
            "'v rušné hlavní městě'; 'hlavním městem světa', NOT 'hlavní město "
            "světa'); past-tense verbs and adjectives must agree with the speaker's "
            "gender (male -> 'řekl/byl', female -> 'řekla/byla'). Output fluent, "
            "grammatically flawless Czech."
        )
    return (
        f"You are a professional subtitle translator and dubbing adapter translating "
        f"from {src} to {tgt}. Rules: translate meaning precisely and naturally; "
        f"preserve each speaker's gender so target grammar agrees with it; keep "
        f"technical terms and established anglicisms natural for the domain; match the "
        f"register and tone; keep each line roughly the same length so it fits its time "
        f"slot; never merge or split lines; translate every id exactly once.{tgt_rules} "
        f"Each 'text' value must contain ONLY the final translated sentence — no "
        f"reasoning, no notes, no commentary, no quotes, no English. Return ONLY JSON "
        f"matching the schema."
    )


def translate_window(window, context, src: str, tgt: str) -> dict[int, str]:
    ctx_txt = ""
    if context:
        ctx_txt = "Already-translated context (for continuity, do NOT re-translate):\n" + \
            "\n".join(f"- {c}" for c in context) + "\n\n"
    payload = [{"id": s.id, "speaker": s.speaker, "gender": s.gender, "text": s.text_src}
               for s in window]
    user = (ctx_txt + "Translate each line's text to " + tgt +
            ". Return JSON {\"translations\":[{\"id\":<int>,\"text\":<str>}]}.\n\n" +
            json.dumps(payload, ensure_ascii=False))
    messages = [{"role": "system", "content": _system_prompt(src, tgt)},
                {"role": "user", "content": user}]
    max_tokens = min(4096, 256 + 256 * len(window))
    for attempt in range(3):
        try:
            content = _chat(messages, max_tokens=max_tokens,
                            temperature=0.2 + 0.1 * attempt)
            data = _extract_json(content)
            src_by_id = {s.id: s.text_src for s in window}
            out = {}
            for t in data.get("translations", []):
                sid = int(t["id"])
                txt = _clean_text(str(t.get("text", "")))
                if not txt or _is_degenerate(txt):
                    continue  # drop -> caller falls back / retries
                if tgt == "cs" and _is_contaminated(src_by_id.get(sid, ""), txt):
                    continue  # leaked reasoning / meta text -> reject
                out[sid] = txt
            if out:
                return out
            print(f"[translate] window attempt {attempt + 1}: empty/degenerate, retrying")
        except (OSError, KeyError, ValueError) as e:  # incl. TimeoutError/URLError
            print(f"[translate] window retry {attempt + 1}: {e}")
        time.sleep(1)
    return {}


_POLISH_PROMPT = (
    "You are a meticulous Czech proofreader producing a broadcast-quality dub. For each "
    "line, correct the grammar of 'text': case (pády), gender/number agreement, the "
    "correct case after prepositions, verb agreement (incl. past-tense gender per the "
    "speaker), word order, and especially the declension of proper nouns and place "
    "names. A 'reference' field may be given (a specialized machine translation that "
    "usually has correct Czech grammar) - use it to pick the correct forms where 'text' "
    "is wrong, but KEEP 'text's meaning, wording and style; do not copy the reference "
    "wholesale. If a line is already correct, return it unchanged. Return ONLY JSON "
    "matching the schema."
)


def polish_window(window) -> dict[int, str]:
    """Second pass: fix Czech grammar of already-translated lines."""
    payload = []
    for s in window:
        item = {"id": s.id, "gender": s.gender, "text": s.text_tgt}
        if s.text_nmt:
            item["reference"] = s.text_nmt
        payload.append(item)
    user = ("Fix the Czech grammar of each line's 'text' (use 'reference' for the correct "
            "forms where given). Return JSON {\"translations\":[{\"id\":<int>,\"text\":<str>}]} "
            "with one entry per id.\n\n" + json.dumps(payload, ensure_ascii=False))
    messages = [{"role": "system", "content": _POLISH_PROMPT},
                {"role": "user", "content": user}]
    max_tokens = min(4096, 256 + 256 * len(window))
    for attempt in range(2):
        try:
            content = _chat(messages, max_tokens=max_tokens, temperature=0.2 + 0.1 * attempt)
            data = _extract_json(content)
            orig = {s.id: s.text_tgt for s in window}
            out = {}
            for t in data.get("translations", []):
                sid = int(t["id"])
                txt = _clean_text(str(t.get("text", "")))
                if txt and not _is_degenerate(txt) and not _is_contaminated(orig.get(sid, ""), txt):
                    out[sid] = txt
            if out:
                return out
        except (OSError, KeyError, ValueError) as e:
            print(f"[polish] window retry {attempt + 1}: {e}")
        time.sleep(1)
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Translate transcript via LM Studio.")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--target", default="cs", help="Target language (default cs).")
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--keep-loaded", action="store_true", help="Do not unload the LLM.")
    ap.add_argument("--no-polish", action="store_true",
                    help="Skip the Czech grammar-polish second pass.")
    ap.add_argument("--polish-only", action="store_true",
                    help="Skip translation; only re-run the grammar polish on existing text.")
    ap.add_argument("--host", default=None,
                    help="LM Studio base URL, e.g. http://192.168.88.111:1234 (default: local).")
    ap.add_argument("--model", default=None, help="Model id to use (default from config).")
    ap.add_argument("--window", type=int, default=None,
                    help="Segments per LLM batch (default: 24 remote / 8 local).")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    proj.target_language = args.target
    src = proj.source_language or "auto"
    if not proj.segments:
        print("[translate] no segments; run transcribe.py first", file=sys.stderr)
        return 2

    global API, MODEL, WINDOW
    if args.host:
        API = args.host.rstrip("/") + "/v1"
    if args.model:
        MODEL = args.model
    remote = not any(h in API for h in ("localhost", "127.0.0.1"))
    WINDOW = args.window or (24 if remote else 8)

    if remote:
        # Remote host (e.g. Mac Studio): the server JIT-loads the model itself;
        # we must not touch it via the local `lms` CLI.
        print(f"[translate] remote LM Studio {API}, model={MODEL}")
    else:
        ctx = args.context_length or vram.suggest_llm_context(LLM_WEIGHTS_GB)
        ensure_server()
        unload_llm()  # ensure a clean VRAM slate before loading
        load_llm(ctx)
    try:
        if not args.polish_only:
            recent: list[str] = []
            missed = []
            for i in range(0, len(proj.segments), WINDOW):
                window = proj.segments[i:i + WINDOW]
                mapping = translate_window(window, recent[-CONTEXT_LINES:], src, args.target)
                for seg in window:
                    if seg.id in mapping:
                        seg.text_tgt = mapping[seg.id]
                    else:
                        missed.append(seg)
                    recent.append(seg.text_tgt or seg.text_src)
                print(f"[translate] {min(i + WINDOW, len(proj.segments))}/{len(proj.segments)} lines")

            # Retry any line the batch missed, one at a time, so English does not leak in.
            for seg in missed:
                seg.text_tgt = translate_window([seg], [], src, args.target).get(seg.id, seg.text_src)
            if missed:
                print(f"[translate] retried {len(missed)} missed line(s) individually")

        if args.target == "cs" and not args.no_polish:
            for i in range(0, len(proj.segments), WINDOW):
                window = proj.segments[i:i + WINDOW]
                fixed = polish_window(window)
                for seg in window:
                    if seg.id in fixed:
                        seg.text_tgt = fixed[seg.id]
                print(f"[polish] {min(i + WINDOW, len(proj.segments))}/{len(proj.segments)} lines")

        # Final sweep over ALL lines: re-translate anything that still looks like
        # leaked reasoning / meta text ("wait, let me retranslate ...").
        if args.target == "cs":
            bad = [s for s in proj.segments if _is_contaminated(s.text_src, s.text_tgt)]
            for s in bad:
                for _ in range(3):
                    m = translate_window([s], [], src, args.target)
                    if s.id in m:
                        s.text_tgt = m[s.id]
                        break
                else:
                    s.text_tgt = s.text_src
            if bad:
                print(f"[translate] contamination sweep re-did {len(bad)} line(s): "
                      f"{[s.id for s in bad]}")

        for seg in proj.segments:
            seg.text_tts = (normalize_text(seg.text_tgt) if args.target == "cs"
                            else seg.text_tgt)
    finally:
        if not remote and not args.keep_loaded:
            unload_llm()

    proj.save(workdir / "project.json")
    print(f"[translate] {src} -> {args.target}, {len(proj.segments)} lines translated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
