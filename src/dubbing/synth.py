"""Phase 3a: synthesize each line with a unique, gender-matched voice per speaker.

Runs in the TTS venv (OmniVoice). For every diarized speaker we ensure a
reference voice: a user-provided clone wav if given, otherwise a distinct
gender/pitch-varied Czech seed generated once and reused for all that speaker's
lines (so the voice stays consistent). Each line is fitted to its subtitle slot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.schema import DubProject, Speaker  # noqa: E402
from tts_service.dub import assemble_track  # noqa: E402
from tts_service.tts import SAMPLE_RATE, CzechTTS  # noqa: E402

# Distinct pitch rotation so multiple speakers of the same gender sound different.
_PITCHES = ["moderate pitch", "low pitch", "high pitch", "very low pitch", "very high pitch"]
_SEED_TEXT = "Dobrý den, toto je ukázka hlasu pro dabing."


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
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    if not proj.segments:
        print("[synth] no segments; run earlier phases first", file=sys.stderr)
        return 2

    target = proj.target_language or "cs"
    tts = CzechTTS(num_step=args.num_step, guidance_scale=args.guidance, language=target)
    refs_dir = workdir / "voices"

    for idx, spk_id in enumerate(sorted(proj.speakers)):
        ensure_speaker_voice(tts, proj.speakers[spk_id], idx, refs_dir, target)

    clips = []
    for seg in proj.segments:
        spk = proj.speakers.get(seg.speaker) or Speaker(id=seg.speaker)
        text = seg.text_tts or seg.text_tgt or seg.text_src
        audio = tts.synthesize(
            text, ref_audio=spk.ref_voice, ref_text=spk.ref_text,
            language=target, duration=max(seg.slot, 0.3), exact_duration=True,
            normalize=False)  # text_tts is already normalized
        max_len = int(round(seg.slot * SAMPLE_RATE))
        if max_len and len(audio) > max_len:
            audio = audio[:max_len]
        clips.append((seg.start, audio))
        print(f"[synth] seg {seg.id} {seg.speaker}/{seg.gender} "
              f"@ {seg.start:.2f}s slot={seg.slot:.2f}s", flush=True)

    total = max(s.end for s in proj.segments)
    track = assemble_track(clips, total)
    out = workdir / "dubbed_vocals.wav"
    sf.write(str(out), track, SAMPLE_RATE)
    proj.save(workdir / "project.json")
    print(f"[synth] wrote {out.name} ({total:.1f}s, {len(proj.speakers)} speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
