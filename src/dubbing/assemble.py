"""Phase 3b: mix dubbed vocals with the original instrumental and mux to video."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.schema import DubProject  # noqa: E402


def to_stereo_sr(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import torch
    import torchaudio

    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    t = t.unsqueeze(0) if t.ndim == 1 else t.T  # -> [channels, samples]
    if sr_in != sr_out:
        t = torchaudio.functional.resample(t, sr_in, sr_out)
    if t.shape[0] == 1:
        t = t.repeat(2, 1)
    return t.T.numpy()  # [samples, 2]


def mux_video(video: str, audio_wav: str, out_path: str) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-i", audio_wav,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "256k", out_path],
        check=True, capture_output=True, text=True,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mix dub + music and mux to video.")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--vocals-gain", type=float, default=1.0)
    ap.add_argument("--music-gain", type=float, default=0.6)
    ap.add_argument("--out", default=None, help="Output video path.")
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    sr = proj.sr

    voc, vsr = sf.read(str(workdir / "dubbed_vocals.wav"))
    voc_st = to_stereo_sr(voc, vsr, sr)

    if proj.audio_instrumental and Path(proj.audio_instrumental).exists():
        inst, isr = sf.read(proj.audio_instrumental)
        inst_st = to_stereo_sr(inst, isr, sr)
    else:
        inst_st = np.zeros_like(voc_st)

    n = max(len(voc_st), len(inst_st))
    voc_st = np.pad(voc_st, ((0, n - len(voc_st)), (0, 0)))
    inst_st = np.pad(inst_st, ((0, n - len(inst_st)), (0, 0)))
    mix = np.clip(args.vocals_gain * voc_st + args.music_gain * inst_st, -1.0, 1.0)

    final_audio = workdir / "final_audio.wav"
    sf.write(str(final_audio), mix, sr)
    print(f"[assemble] mixed audio -> {final_audio.name} ({n / sr:.1f}s)")

    result = {"final_audio": str(final_audio), "final_video": None}
    if proj.source_video and Path(proj.source_video).exists():
        out = args.out or str(workdir / (Path(proj.source_video).stem + "_dub.mp4"))
        mux_video(proj.source_video, str(final_audio), out)
        result["final_video"] = out
        print(f"[assemble] dubbed video -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
