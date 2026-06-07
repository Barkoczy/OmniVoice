"""CLI: synthesize a single piece of Czech text to a WAV file.

Examples:
  python scripts/say.py "Ahoj světe, dnes je krásný den."
  python scripts/say.py --file examples/demo_cs.txt --out output/demo.wav
  python scripts/say.py "Klonovaný hlas." --ref refs/my_voice.wav --ref-text "..."
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tts_service.tts import CzechTTS, SAMPLE_RATE  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Czech text-to-speech via OmniVoice.")
    p.add_argument("text", nargs="?", help="Text to speak (or use --file / stdin).")
    p.add_argument("--file", type=Path, help="Read text from this file.")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "say.wav",
                   help="Output WAV path (default: output/say.wav).")
    p.add_argument("--ref", help="Reference voice WAV (3-10s) to clone.")
    p.add_argument("--ref-text", help="Transcript of the reference (else auto-ASR).")
    p.add_argument("--instruct", help="Voice design attributes, e.g. 'male, low pitch'.")
    p.add_argument("--language", default="cs", help="Language id (default: cs; '' = auto).")
    p.add_argument("--num-step", type=int, default=48, help="Diffusion steps (default 48).")
    p.add_argument("--guidance", type=float, default=2.0, help="Guidance scale (default 2.0).")
    p.add_argument("--speed", type=float, default=None, help="Speed factor (>1 faster).")
    p.add_argument("--duration", type=float, default=None, help="Force output length (s).")
    p.add_argument("--no-normalize", action="store_true", help="Disable Czech normalization.")
    p.add_argument("--device", default="cuda:0", help="Torch device (cuda:0/cpu/mps/xpu).")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        print("No input text provided.", file=sys.stderr)
        return 2

    tts = CzechTTS(
        device=args.device, dtype=args.dtype,
        num_step=args.num_step, guidance_scale=args.guidance,
        language=args.language or None,
    )
    out = tts.to_file(
        text, args.out,
        ref_audio=args.ref, ref_text=args.ref_text, instruct=args.instruct,
        speed=args.speed, duration=args.duration,
        normalize=not args.no_normalize,
        exact_duration=args.duration is not None,
    )
    import soundfile as sf
    info = sf.info(str(out))
    print(f"Wrote {out}  ({info.duration:.2f}s @ {SAMPLE_RATE} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
