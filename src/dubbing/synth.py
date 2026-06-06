"""Phase 3a: synthesize each line with a unique, gender-matched voice per speaker.

Runs in the TTS venv (OmniVoice). For every diarized speaker we ensure a
reference voice: a user-provided clone wav if given, otherwise a distinct
gender/pitch-varied Czech seed generated once and reused for all that speaker's
lines (so the voice stays consistent). Each line is fitted to its subtitle slot.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.schema import DubProject, Speaker  # noqa: E402
from tts_service.dub import assemble_track  # noqa: E402
from tts_service.tts import SAMPLE_RATE, CzechTTS  # noqa: E402

# Distinct pitch rotation so multiple speakers of the same gender sound different.
_PITCHES = ["moderate pitch", "low pitch", "high pitch", "very low pitch", "very high pitch"]
_SEED_TEXT = "Dobrý den, toto je ukázka hlasu pro dabing."


def _time_stretch(audio, factor: float):
    """Pitch-preserving time-stretch of mono audio via ffmpeg atempo (>1 = faster).

    Operates on the already-rendered (complete) audio, so no words are ever lost -
    unlike re-synthesizing with a speed factor, which can drop or slur words.
    """
    if abs(factor - 1.0) < 0.02 or not shutil.which("ffmpeg"):
        return audio
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(ip), audio, SAMPLE_RATE)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ip), "-filter:a", f"atempo={factor:.4f}", str(op)],
            check=True, capture_output=True,
        )
        out, _ = sf.read(str(op))
    return np.asarray(out, dtype="float32")


def ensure_speaker_voice(tts: CzechTTS, spk: Speaker, idx: int, refs_dir: Path,
                         language: str = "cs") -> None:
    if spk.ref_voice and Path(spk.ref_voice).exists():
        return
    gender = "female" if spk.gender == "female" else "male"
    instruct = f"{gender}, {_PITCHES[idx % len(_PITCHES)]}"
    audio = tts.synthesize(_SEED_TEXT, instruct=instruct, language=language,
                           normalize=False, num_step=48)
    refs_dir.mkdir(parents=True, exist_ok=True)
    path = refs_dir / f"voice_{spk.id}.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    spk.ref_voice = str(path)
    spk.ref_text = _SEED_TEXT
    print(f"[synth] generated voice for {spk.id} ({instruct}) -> {path.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Multi-speaker TTS fitted to slots.")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--num-step", type=int, default=48)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--narrator", default=None,
                    help="Reference wav (3-10s) applied to ALL speakers (single voice).")
    ap.add_argument("--voice", action="append", default=[], metavar="ID=PATH",
                    help="Per-speaker reference wav, e.g. SPEAKER_00=refs/narrator.wav (repeatable).")
    ap.add_argument("--ref-text", default=None,
                    help="Transcript of the reference(s); else Whisper auto-transcribes.")
    ap.add_argument("--max-speed", type=float, default=1.4,
                    help="Max speed-up used to fit a line into its slot (default 1.4).")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    if not proj.segments:
        print("[synth] no segments; run earlier phases first", file=sys.stderr)
        return 2

    target = proj.target_language or "cs"

    # Apply user-provided reference voices (override auto-generation).
    if args.narrator:
        for spk in proj.speakers.values():
            spk.ref_voice, spk.ref_text = args.narrator, args.ref_text
    for item in args.voice:
        sid, _, path = item.partition("=")
        if sid in proj.speakers and path:
            proj.speakers[sid].ref_voice = path
            proj.speakers[sid].ref_text = args.ref_text
        else:
            print(f"[synth] ignoring --voice {item!r}; known speakers: {list(proj.speakers)}")

    tts = CzechTTS(num_step=args.num_step, guidance_scale=args.guidance, language=target)
    refs_dir = workdir / "voices"

    for idx, spk_id in enumerate(sorted(proj.speakers)):
        ensure_speaker_voice(tts, proj.speakers[spk_id], idx, refs_dir, target)

    clips = []
    for seg in proj.segments:
        spk = proj.speakers.get(seg.speaker) or Speaker(id=seg.speaker)
        text = seg.text_tts or seg.text_tgt or seg.text_src
        # Render at natural pace so the whole sentence is always spoken in full...
        audio = tts.synthesize(
            text, ref_audio=spk.ref_voice, ref_text=spk.ref_text,
            language=target, normalize=False)
        nat = len(audio) / SAMPLE_RATE
        # ...then time-stretch the finished audio to fit its slot (keeps every word,
        # pitch-preserving). Skip very short clips to avoid stretch artifacts.
        if seg.slot >= 0.8 and nat > seg.slot * 1.05:
            factor = min(nat / seg.slot, args.max_speed)
            audio = _time_stretch(audio, factor)
        clips.append((seg.start, audio))
        print(f"[synth] seg {seg.id} {seg.speaker}/{seg.gender} @ {seg.start:.2f}s "
              f"slot={seg.slot:.2f}s out={len(audio)/SAMPLE_RATE:.2f}s", flush=True)

    total = max(start + len(a) / SAMPLE_RATE for start, a in clips)
    track = assemble_track(clips, total)
    out = workdir / "dubbed_vocals.wav"
    sf.write(str(out), track, SAMPLE_RATE)
    proj.save(workdir / "project.json")
    print(f"[synth] wrote {out.name} ({total:.1f}s, {len(proj.speakers)} speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
