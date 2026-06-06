"""CLI: dub a video/audio from an .srt subtitle file using a cloned Czech voice.

Examples:
  # Audio-only (aligned WAV track):
  python scripts/dub_video.py --srt examples/example.srt --ref refs/auto_cs.wav \
      --ref-text "Dobrý den, vítejte ..." --out-audio output/dub.wav

  # Full video dubbing (replace original audio):
  python scripts/dub_video.py --srt subs.srt --ref refs/voice.wav \
      --video movie.mp4 --out-video output/movie_cs.mp4

  # Keep the original audio quietly in the background (ducking):
  python scripts/dub_video.py --srt subs.srt --ref refs/voice.wav --video movie.mp4 \
      --keep-original --original-volume 0.2
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tts_service.tts import CzechTTS  # noqa: E402
from tts_service.dub import dub_srt  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Czech SRT dubbing via OmniVoice + ffmpeg.")
    p.add_argument("--srt", type=Path, required=True, help="Subtitle file (.srt).")
    p.add_argument("--ref", help="Reference Czech voice WAV (3-10s) to clone.")
    p.add_argument("--ref-text", help="Transcript of the reference (else auto-ASR).")
    p.add_argument("--instruct", help="Voice design instead of cloning (EN/CN only).")
    p.add_argument("--out-audio", type=Path, default=ROOT / "output" / "dub.wav",
                   help="Output aligned WAV (default: output/dub.wav).")
    p.add_argument("--video", type=Path, help="Optional source video to dub.")
    p.add_argument("--out-video", type=Path, help="Output video path (default: <out-audio>.mp4).")
    p.add_argument("--fit", choices=["duration", "natural"], default="duration",
                   help="'duration' fits each line to its slot; 'natural' = free pace.")
    p.add_argument("--language", default="cs", help="Language id (default cs).")
    p.add_argument("--num-step", type=int, default=48, help="Diffusion steps (default 48).")
    p.add_argument("--guidance", type=float, default=2.0, help="Guidance scale (default 2.0).")
    p.add_argument("--no-normalize", action="store_true", help="Disable Czech normalization.")
    p.add_argument("--keep-original", action="store_true",
                   help="Mix original audio under the dub instead of replacing it.")
    p.add_argument("--original-volume", type=float, default=0.25,
                   help="Original audio volume when --keep-original (default 0.25).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tts = CzechTTS(num_step=args.num_step, guidance_scale=args.guidance,
                   language=args.language)
    result = dub_srt(
        srt_path=args.srt,
        ref_audio=args.ref,
        out_audio=args.out_audio,
        tts=tts,
        ref_text=args.ref_text,
        instruct=args.instruct,
        language=args.language,
        fit=args.fit,
        normalize=not args.no_normalize,
        video=args.video,
        out_video=args.out_video,
        keep_original=args.keep_original,
        original_volume=args.original_volume,
    )
    print("\nDone:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
