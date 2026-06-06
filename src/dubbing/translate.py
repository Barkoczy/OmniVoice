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

WINDOW = 15            # segments per LLM call
CONTEXT_LINES = 3      # preceding lines passed for continuity

_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}


def _lms(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["lms", *args], capture_output=True, text=True, check=check)


def ensure_server() -> None:
    try:
        _lms("server", "start")
    except subprocess.CalledProcessError as e:
        print(f"[translate] lms server start: {e.stderr or e.stdout}")


def load_llm(context_length: int) -> None:
    print(f"[translate] loading {LMSTUDIO_MODEL} (ctx={context_length}, gpu=max)")
    _lms("load", LMSTUDIO_MODEL, "--gpu", "max",
         "--context-length", str(context_length), "--identifier", "dub-llm", "--yes")


def unload_llm() -> None:
    _lms("unload", "--all", check=False)


def _chat(messages: list[dict], max_tokens: int = 3000, temperature: float = 0.2) -> str:
    body = {
        "model": LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "translations", "schema": _SCHEMA, "strict": True},
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(LMSTUDIO_API + "/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]


def _system_prompt(src: str, tgt: str) -> str:
    tgt_rules = ""
    if tgt == "cs":
        tgt_rules = (
            " The target is Czech: use correct grammatical gender for past-tense "
            "verbs and adjectives based on each line's speaker gender (male -> "
            "'řekl/byl rád', female -> 'řekla/byla ráda'). Keep numbers as words "
            "is NOT required here (a later step handles it)."
        )
    return (
        f"You are a professional subtitle translator and dubbing adapter translating "
        f"from {src} to {tgt}. Rules: translate meaning precisely and naturally; "
        f"preserve each speaker's gender so target grammar agrees with it; keep "
        f"technical terms and established anglicisms natural for the domain; match the "
        f"register and tone; keep each line roughly the same length so it fits its time "
        f"slot; never merge or split lines; translate every id exactly once.{tgt_rules} "
        f"Return ONLY JSON matching the schema."
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
    for attempt in range(3):
        try:
            content = _chat(messages)
            data = json.loads(content)
            return {int(t["id"]): t["text"].strip() for t in data["translations"]}
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            print(f"[translate] window retry {attempt + 1}: {e}")
            time.sleep(2)
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Translate transcript via LM Studio.")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--target", default="cs", help="Target language (default cs).")
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--keep-loaded", action="store_true", help="Do not unload the LLM.")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    proj.target_language = args.target
    src = proj.source_language or "auto"
    if not proj.segments:
        print("[translate] no segments; run transcribe.py first", file=sys.stderr)
        return 2

    ctx = args.context_length or vram.suggest_llm_context(LLM_WEIGHTS_GB)
    ensure_server()
    load_llm(ctx)
    try:
        recent: list[str] = []
        for i in range(0, len(proj.segments), WINDOW):
            window = proj.segments[i:i + WINDOW]
            mapping = translate_window(window, recent[-CONTEXT_LINES:], src, args.target)
            for seg in window:
                seg.text_tgt = mapping.get(seg.id, seg.text_src)
                seg.text_tts = normalize_text(seg.text_tgt) if args.target == "cs" else seg.text_tgt
                recent.append(seg.text_tgt)
            print(f"[translate] {min(i + WINDOW, len(proj.segments))}/{len(proj.segments)} lines")
    finally:
        if not args.keep_loaded:
            unload_llm()

    proj.save(workdir / "project.json")
    print(f"[translate] {src} -> {args.target}, {len(proj.segments)} lines translated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
