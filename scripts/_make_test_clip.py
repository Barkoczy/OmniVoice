"""Make a synthetic speech+music clip to test source separation objectively.

Mixes the Czech reference speech with a synthetic musical bed (clearly distinct
from speech) so we can measure how well separation recovers each stem.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "output"
out.mkdir(exist_ok=True)

sp, sr = sf.read(str(ROOT / "refs" / "auto_cs.wav"))
sp = np.asarray(sp, dtype=np.float32)
if sp.ndim > 1:
    sp = sp.mean(axis=1)
SR = 44100
sp44 = torchaudio.functional.resample(torch.from_numpy(sp), sr, SR).numpy()

dur = len(sp44) / SR + 1.0
n = int(dur * SR)
t = np.arange(n) / SR

# Musical bed: tremolo arpeggio of an A-major chord + a bass pulse.
freqs = [220.0, 277.18, 329.63, 440.0]
music = np.zeros(n, dtype=np.float32)
for k, f in enumerate(freqs):
    env = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t + k))
    music += (0.25 * env * np.sin(2 * np.pi * f * t)).astype(np.float32)
music += 0.2 * np.sin(2 * np.pi * 110.0 * t).astype(np.float32)
music *= 0.5
music_st = np.stack([music, np.roll(music, 50)], axis=1)

sp_pad = np.zeros(n, dtype=np.float32)
sp_pad[: len(sp44)] = sp44
sp_st = np.stack([sp_pad, sp_pad], axis=1)

mix = np.clip(0.8 * sp_st + 0.4 * music_st, -1.0, 1.0)

sf.write(str(out / "test_speech44.wav"), sp_st, SR)
sf.write(str(out / "test_music44.wav"), music_st, SR)
sf.write(str(out / "test_mix.wav"), mix, SR)
print(f"wrote output/test_mix.wav shape={mix.shape} dur={dur:.2f}s")
