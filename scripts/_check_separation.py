"""Objectively check separation quality on the synthetic test clip.

High vocals~speech and instrumental~music correlations (and low cross terms)
mean the separator cleanly split speech from music.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "output"


def mono(p: Path):
    a, sr = sf.read(str(p))
    a = np.asarray(a, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a, sr


def corr(a, b):
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


sp, _ = mono(out / "test_speech44.wav")
mu, _ = mono(out / "test_music44.wav")
voc, _ = mono(out / "dubtest" / "audio" / "vocals.wav")
ins, _ = mono(out / "dubtest" / "audio" / "instrumental.wav")

print(f"vocals ~ speech : {corr(voc, sp):+.3f}  (want HIGH)")
print(f"vocals ~ music  : {corr(voc, mu):+.3f}  (want low)")
print(f"instrum ~ music : {corr(ins, mu):+.3f}  (want HIGH)")
print(f"instrum ~ speech: {corr(ins, sp):+.3f}  (want low)")
