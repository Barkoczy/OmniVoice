"""Phase 1 consensus: secondary ASR via NVIDIA Parakeet-TDT v3 (runs in .venv-parakeet).

Transcribes the vocals stem with Parakeet, then for each WhisperX segment records
the overlapping Parakeet text and a similarity score (``asr_agreement``). Low
agreement flags a segment for review; the primary transcription stays WhisperX.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.config import PARAKEET_MODEL  # noqa: E402
from dubbing.schema import DubProject  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def _to_16k_mono(src: str, dst: str) -> None:
    import torch
    import torchaudio

    data, sr = sf.read(src)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    t = torch.from_numpy(data)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    sf.write(dst, t.numpy(), 16000)


def transcribe_parakeet(wav16k: str):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL)
    out = model.transcribe([wav16k], timestamps=True)
    hyp = out[0]
    words = []
    try:
        for w in hyp.timestamp["word"]:
            words.append((float(w["start"]), float(w["end"]), str(w["word"])))
    except Exception:  # noqa: BLE001
        pass
    full = getattr(hyp, "text", "") or ""
    return words, full


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Parakeet secondary-ASR consensus.")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    if not proj.audio_vocals or not Path(proj.audio_vocals).exists():
        print("[parakeet] no vocals stem; run separate.py first", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        wav16 = str(Path(td) / "voc16.wav")
        _to_16k_mono(proj.audio_vocals, wav16)
        words, full = transcribe_parakeet(wav16)

    for seg in proj.segments:
        ptext = " ".join(w for (s, e, w) in words if e > seg.start and s < seg.end).strip()
        if not ptext:
            ptext = full if not words else ""
        seg.text_asr2 = ptext
        if ptext:
            seg.asr_agreement = round(
                difflib.SequenceMatcher(None, _norm(seg.text_src), _norm(ptext)).ratio(), 3)

    proj.save(workdir / "project.json")
    low = [s.id for s in proj.segments if s.asr_agreement < 0.6]
    print(f"[parakeet] consensus on {len(proj.segments)} segments; "
          f"low-agreement (<0.6): {low}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
