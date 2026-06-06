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
    args = ap.parse_args(argv)

    workdir = args.workdir or str(ROOT / "output" / Path(args.video).stem)
    steps = args.steps.split(",") if args.steps else None
    run_pipeline(
        args.video, workdir, source=args.source, target=args.target,
        separation=args.separation, min_speakers=args.min_speakers,
        max_speakers=args.max_speakers, steps=steps,
    )
    print(f"\nPipeline finished. Work dir: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
