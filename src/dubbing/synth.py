"""Phase 3a: synthesize each line in the ORIGINAL speaker's voice, paced naturally.

By default each diarized speaker's reference voice is extracted from the original
separated vocals, so the dub keeps the original narrator's timbre/melody
(cross-lingual cloning). Lines are paced into their subtitle slot by OmniVoice's
own (natural) duration conditioning - never by external time-stretching, which
sounded robotic - and are placed so two lines never overlap (overlapping lines
were the "second voice" heard in the background).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.schema import DubProject, Speaker  # noqa: E402
from tts_service.dub import assemble_track  # noqa: E402
from tts_service.tts import SAMPLE_RATE, CzechTTS  # noqa: E402

_PITCHES = ["moderate pitch", "low pitch", "high pitch", "very low pitch", "very high pitch"]
_SEED_TEXT = "Dobrý den, toto je ukázka hlasu pro dabing."


def extract_speaker_ref(proj, spk_id, refs_dir, lo=3.0, hi=7.0):
    """Reference clip for a speaker from the ORIGINAL vocals: ONE clean, short
    utterance (a multi-sentence clip makes OmniVoice echo the reference into the
    output). Returns (wav_path, ref_text) with the exactly-aligned source text.
    """
    vocals = proj.audio_vocals
    if not vocals or not Path(vocals).exists():
        return None, None
    data, sr = sf.read(vocals)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    cand = [s for s in proj.segments if s.speaker == spk_id and lo <= s.slot <= hi]
    if not cand:
        cand = [s for s in proj.segments if s.speaker == spk_id and s.slot >= 2.0]
    if not cand:
        return None, None
    seg = max(cand, key=lambda s: s.slot)  # longest single utterance in range
    clip = data[int(seg.start * sr):int(seg.end * sr)]
    if len(clip) == 0:
        return None, None
    refs_dir.mkdir(parents=True, exist_ok=True)
    path = refs_dir / f"orig_{spk_id}.wav"
    sf.write(str(path), clip, sr)
    return str(path), seg.text_src.strip()


def auto_voice(tts, spk, idx, refs_dir, language="cs"):
    """Fallback: a distinct synthetic gender/pitch-varied seed voice."""
    gender = "female" if spk.gender == "female" else "male"
    instruct = f"{gender}, {_PITCHES[idx % len(_PITCHES)]}"
    audio = tts.synthesize(_SEED_TEXT, instruct=instruct, language=language,
                           normalize=False, num_step=48)
    refs_dir.mkdir(parents=True, exist_ok=True)
    path = refs_dir / f"voice_{spk.id}.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    return str(path), _SEED_TEXT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Multi-speaker TTS in the original voice.")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--num-step", type=int, default=64)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--narrator", default=None,
                    help="Reference wav for ALL speakers (overrides original-voice clone).")
    ap.add_argument("--voice", action="append", default=[], metavar="ID=PATH",
                    help="Per-speaker reference wav (repeatable; overrides original clone).")
    ap.add_argument("--ref-text", default=None, help="Transcript for --narrator/--voice refs.")
    ap.add_argument("--auto-voice", action="store_true",
                    help="Force synthetic voices (ignore any preset reference).")
    ap.add_argument("--clone-original", action="store_true",
                    help="Clone each speaker from the ORIGINAL audio (can echo a foreign "
                         "reference into the output; off by default).")
    ap.add_argument("--fit", action="store_true",
                    help="Fit each line into its subtitle slot via OmniVoice duration "
                         "(can drop word-starts; OFF by default - natural pace is reliable).")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    if not proj.segments:
        print("[synth] no segments; run earlier phases first", file=sys.stderr)
        return 2

    target = proj.target_language or "cs"
    refs_dir = workdir / "voices"

    user_set = set()
    if args.narrator:
        for spk in proj.speakers.values():
            spk.ref_voice, spk.ref_text = args.narrator, args.ref_text
            user_set.add(spk.id)
    for item in args.voice:
        sid, _, path = item.partition("=")
        if sid in proj.speakers and path:
            proj.speakers[sid].ref_voice = path
            proj.speakers[sid].ref_text = args.ref_text
            user_set.add(sid)

    tts = CzechTTS(num_step=args.num_step, guidance_scale=args.guidance, language=target)

    # Reference per speaker: user override > preset (e.g. a Czech sample set on the
    # project) > clone-from-original (opt-in) > synthetic. Cloning the original is
    # OFF by default because a foreign-language reference can echo into the output.
    for idx, sid in enumerate(sorted(proj.speakers)):
        spk = proj.speakers[sid]
        if sid in user_set:
            src = "user"
        elif args.auto_voice:
            spk.ref_voice, spk.ref_text = auto_voice(tts, spk, idx, refs_dir, target)
            src = "auto"
        elif args.clone_original:
            rv, rt = extract_speaker_ref(proj, sid, refs_dir)
            if rv:
                spk.ref_voice, spk.ref_text, src = rv, rt, "original"
            else:
                spk.ref_voice, spk.ref_text = auto_voice(tts, spk, idx, refs_dir, target)
                src = "auto"
        elif spk.ref_voice and Path(spk.ref_voice).exists():
            src = "preset"
        else:
            spk.ref_voice, spk.ref_text = auto_voice(tts, spk, idx, refs_dir, target)
            src = "auto"
        print(f"[synth] {sid} ({spk.gender}) voice={src} -> {Path(spk.ref_voice).name}")

    clips = []
    prev_end = 0.0
    for seg in proj.segments:
        spk = proj.speakers.get(seg.speaker) or Speaker(id=seg.speaker)
        text = seg.text_tts or seg.text_tgt or seg.text_src
        duration = seg.slot if (args.fit and seg.slot >= 1.0) else None  # natural pace by default
        audio = tts.synthesize(
            text, ref_audio=spk.ref_voice, ref_text=spk.ref_text,
            language=target, duration=duration, normalize=False)
        start = max(seg.start, prev_end)  # never overlap the previous line
        prev_end = start + len(audio) / SAMPLE_RATE
        clips.append((start, audio))
        print(f"[synth] seg {seg.id} {seg.speaker}/{seg.gender} @ {start:.2f}s "
              f"slot={seg.slot:.2f}s out={len(audio)/SAMPLE_RATE:.2f}s", flush=True)

    total = max(start + len(a) / SAMPLE_RATE for start, a in clips)
    track = assemble_track(clips, total)
    peak = float(np.max(np.abs(track))) or 1.0
    track = (track / peak) * 0.89  # consistent level, no loudnorm pumping
    out = workdir / "dubbed_vocals.wav"
    sf.write(str(out), track, SAMPLE_RATE)
    proj.save(workdir / "project.json")
    print(f"[synth] wrote {out.name} ({total:.1f}s, {len(proj.speakers)} speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
