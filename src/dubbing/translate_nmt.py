"""Phase 2 (offline): translate with a local open-source NMT model.

Fully offline / open-source, no cloud, no registration. A dedicated EN->CS NMT
(Opus-MT by default) declines Czech proper nouns and grammar far more reliably
than a general LLM, generically for any video. NLLB-200 is also supported.

Runs in the analysis venv (transformers + sentencepiece). Output is normalized
for Czech TTS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.schema import DubProject  # noqa: E402
from tts_service.cz_normalize import normalize_text  # noqa: E402

# NLLB uses FLORES-200 language codes.
_NLLB_LANG = {
    "en": "eng_Latn", "cs": "ces_Latn", "de": "deu_Latn", "fr": "fra_Latn",
    "es": "spa_Latn", "pl": "pol_Latn", "sk": "slk_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "it": "ita_Latn", "pt": "por_Latn",
}


def build(model: str, src: str, tgt: str):
    from transformers import pipeline
    if "nllb" in model.lower():
        return pipeline("translation", model=model, device=0,
                        src_lang=_NLLB_LANG.get(src, "eng_Latn"),
                        tgt_lang=_NLLB_LANG.get(tgt, "ces_Latn"))
    return pipeline("translation", model=model, device=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline NMT translation (EN->CS etc.).")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--source", default=None, help="Source language (default: from project).")
    ap.add_argument("--target", default="cs")
    ap.add_argument("--model", default=None,
                    help="HF model id; default Helsinki-NLP/opus-mt-<src>-<tgt>. "
                         "Use e.g. facebook/nllb-200-distilled-1.3B for NLLB.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--field", default="text_tgt", choices=["text_tgt", "text_nmt"],
                    help="Where to store output: text_tgt (standalone) or text_nmt "
                         "(grammar reference for the hybrid refine step).")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    if not proj.segments:
        print("[nmt] no segments; run earlier phases first", file=sys.stderr)
        return 2

    src = args.source or proj.source_language or "en"
    tgt = args.target
    proj.target_language = tgt
    model = args.model or f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    print(f"[nmt] {src} -> {tgt} via {model}")

    translator = build(model, src, tgt)
    texts = [s.text_src for s in proj.segments]
    outs = translator(texts, max_length=512, batch_size=args.batch_size, truncation=True)

    for seg, o in zip(proj.segments, outs):
        txt = o["translation_text"].strip()
        setattr(seg, args.field, txt)
        if args.field == "text_tgt":
            seg.text_tts = normalize_text(txt) if tgt == "cs" else txt

    proj.save(workdir / "project.json")
    print(f"[nmt] translated {len(proj.segments)} segments -> {args.field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
