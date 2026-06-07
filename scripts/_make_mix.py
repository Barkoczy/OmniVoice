"""Mix a speech wav with a synthetic musical bed (test input for the pipeline)."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--speech", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--music-gain", type=float, default=0.4)
args = ap.parse_args()

SR = 44100
sp, sr = sf.read(args.speech)
sp = np.asarray(sp, dtype=np.float32)
if sp.ndim > 1:
    sp = sp.mean(axis=1)
sp = torchaudio.functional.resample(torch.from_numpy(sp), sr, SR).numpy()

n = len(sp) + SR  # +1s tail
t = np.arange(n) / SR
music = np.zeros(n, dtype=np.float32)
for k, f in enumerate([196.0, 246.94, 293.66, 392.0]):  # G major-ish
    env = 0.5 * (1 + np.sin(2 * np.pi * 0.4 * t + k))
    music += (0.25 * env * np.sin(2 * np.pi * f * t)).astype(np.float32)
music += 0.2 * np.sin(2 * np.pi * 98.0 * t).astype(np.float32)
music *= 0.5
music_st = np.stack([music, np.roll(music, 40)], axis=1)

sp_pad = np.zeros(n, dtype=np.float32)
sp_pad[: len(sp)] = sp
sp_st = np.stack([sp_pad, sp_pad], axis=1)
mix = np.clip(0.8 * sp_st + args.music_gain * music_st, -1.0, 1.0)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
sf.write(args.out, mix, SR)
print(f"wrote {args.out} shape={mix.shape} dur={n / SR:.2f}s")
