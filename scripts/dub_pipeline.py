"""CLI: full video dubbing pipeline (any language -> any language).

Example:
  python scripts/dub_pipeline.py --video movie.mp4 --source en --target cs
  python scripts/dub_pipeline.py --video clip.mkv --target cs --separation ensemble \
      --min-speakers 2 --max-speakers 4

Runs: separate -> transcribe(+diarize+gender) -> translate(LM Studio) ->
multi-speaker TTS -> mix + mux. Each phase runs in its own subprocess so GPU
memory is released between phases.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dubbing.orchestrator import run_pipeline  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end video dubbing pipeline.")
    ap.add_argument("--video", required=True, help="Input video or audio file.")
    ap.add_argument("--workdir", default=None, help="Work dir (default output/<name>).")
    ap.add_argument("--source", default=None, help="Source language (default auto).")
    ap.add_argument("--target", default="cs", help="Target language (default cs).")
    ap.add_argument("--separation", default="ensemble",
                    choices=["demucs", "roformer", "ensemble"])
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--steps", default=None,
                    help="Comma subset of: separate,transcribe,consensus,translate,synth,assemble")
    ap.add_argument("--narrator", default=None,
                    help="Reference wav (3-10s) applied to ALL speakers (single-voice dub).")
    ap.add_argument("--voice", action="append", default=[], metavar="ID=PATH",
                    help="Per-speaker reference wav, e.g. SPEAKER_00=refs/voice.wav (repeatable).")
    ap.add_argument("--no-polish", action="store_true",
                    help="Skip the Czech grammar-polish second pass.")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible LLM base URL ending in /v1 (default from .env).")
    ap.add_argument("--api-key", default=None,
                    help="Bearer API key for the LLM backend (default from .env).")
    ap.add_argument("--model", default=None, help="LLM translation model id (overrides default).")
    ap.add_argument("--translator", choices=["hybrid", "nmt", "llm"], default="hybrid",
                    help="hybrid = LLM (context) + offline NMT grammar reference (default); "
                         "nmt = offline NMT only; llm = LLM only.")
    ap.add_argument("--nmt-model", default=None,
                    help="Offline NMT model id (default Helsinki-NLP/opus-mt-<src>-<tgt>).")
    args = ap.parse_args(argv)

    workdir = args.workdir or str(ROOT / "output" / Path(args.video).stem)
    steps = args.steps.split(",") if args.steps else None
    run_pipeline(
        args.video, workdir, source=args.source, target=args.target,
        separation=args.separation, min_speakers=args.min_speakers,
        max_speakers=args.max_speakers, steps=steps,
        narrator=args.narrator, voices=args.voice, polish=not args.no_polish,
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        translator=args.translator, nmt_model=args.nmt_model,
    )
    print(f"\nPipeline finished. Work dir: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
