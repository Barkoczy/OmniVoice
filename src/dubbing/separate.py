"""Phase 1a: extract audio and separate vocals (speech) from instrumental.

Runs in the analysis venv. Produces a clean vocals stem for ASR/diarization and
an instrumental stem (music + SFX) that is mixed back under the dub at the end.

Quality tiers:
- demucs    : Demucs htdemucs_ft (fast, great).
- roformer  : UVR BS-Roformer via the `audio-separator` package (highest SDR).
- ensemble  : average the vocals/instrumental of both (default, best quality).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.config import (DEMUCS_MODEL, ROFORMER_MODEL, SEP_SR,  # noqa: E402
                            VENV_ROFORMER)
from dubbing.schema import DubProject  # noqa: E402

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".ts"}


def extract_audio(src: str | Path, out_wav: Path, sr: int = SEP_SR) -> Path:
    """Extract a stereo PCM wav from a video/audio file via ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", str(sr),
         "-c:a", "pcm_s16le", str(out_wav)],
        check=True, capture_output=True, text=True,
    )
    return out_wav


def separate_demucs(in_wav: Path, outdir: Path, device: str = "cuda"
                    ) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (vocals, instrumental, sr) using the Demucs CLI (two-stems)."""
    sub = outdir / "_demucs"
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-n", DEMUCS_MODEL, "-o", str(sub), "--device", device, str(in_wav)]
    subprocess.run(cmd, check=True)
    stem_dir = sub / DEMUCS_MODEL / in_wav.stem
    v, sr = sf.read(str(stem_dir / "vocals.wav"))
    i, _ = sf.read(str(stem_dir / "no_vocals.wav"))
    return np.asarray(v, dtype=np.float32), np.asarray(i, dtype=np.float32), int(sr)


def separate_roformer(in_wav: Path, outdir: Path) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (vocals, instrumental, sr) via the roformer subprocess, or None.

    Runs in the isolated .venv-roformer; degrades gracefully (returns None, so the
    ensemble falls back to Demucs alone) if that env or the model is unavailable.
    """
    rof_out = outdir / "_roformer"
    rof_out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "roformer.py"
    try:
        subprocess.run(
            [str(VENV_ROFORMER), str(script), "--input", str(in_wav),
             "--outdir", str(rof_out), "--model", ROFORMER_MODEL],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"[separate] roformer unavailable ({e}); using Demucs only")
        return None
    voc, inst = rof_out / "vocals.wav", rof_out / "instrumental.wav"
    if not voc.exists() or not inst.exists():
        return None
    v, sr = sf.read(str(voc))
    i, _ = sf.read(str(inst))
    return np.asarray(v, dtype=np.float32), np.asarray(i, dtype=np.float32), int(sr)


def _align_mean(arrays: list[np.ndarray]) -> np.ndarray:
    n = min(a.shape[0] for a in arrays)
    arrays = [a[:n] for a in arrays]
    return np.mean(np.stack(arrays, axis=0), axis=0)


def separate(in_wav: Path, outdir: Path, models=("demucs", "roformer")
             ) -> tuple[Path, Path, int]:
    """Run the chosen separators, ensemble them, and write vocals/instrumental."""
    outdir.mkdir(parents=True, exist_ok=True)
    voc_list, inst_list, sr = [], [], SEP_SR

    if "demucs" in models:
        v, i, sr = separate_demucs(in_wav, outdir)
        voc_list.append(v)
        inst_list.append(i)
    if "roformer" in models:
        res = separate_roformer(in_wav, outdir / "_roformer")
        if res is not None:
            v, i, rsr = res
            if rsr == sr or not voc_list:
                voc_list.append(v)
                inst_list.append(i)
                sr = rsr
            else:
                print(f"[separate] roformer sr {rsr} != {sr}; skipping in ensemble")
        else:
            print("[separate] roformer unavailable; using demucs only")

    if not voc_list:
        raise RuntimeError("No separation model produced output.")

    vocals = _align_mean(voc_list)
    instrumental = _align_mean(inst_list)
    voc_path = outdir / "vocals.wav"
    inst_path = outdir / "instrumental.wav"
    sf.write(str(voc_path), vocals, sr)
    sf.write(str(inst_path), instrumental, sr)
    return voc_path, inst_path, sr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Separate vocals from instrumental.")
    ap.add_argument("--input", required=True, help="Video or audio file.")
    ap.add_argument("--workdir", required=True, help="Project work directory.")
    ap.add_argument("--models", default="ensemble",
                    choices=["demucs", "roformer", "ensemble"])
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    proj_path = workdir / "project.json"
    proj = DubProject.load(proj_path) if proj_path.exists() else DubProject(workdir=str(workdir))

    src = Path(args.input)
    audio_dir = workdir / "audio"
    if src.suffix.lower() in VIDEO_EXTS:
        proj.source_video = str(src)
        full = extract_audio(src, audio_dir / "full.wav")
    else:
        full = audio_dir / "full.wav"
        full.parent.mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(str(src))
        sf.write(str(full), data, sr)
    proj.audio_full = str(full)

    models = ("demucs", "roformer") if args.models == "ensemble" else (args.models,)
    voc, inst, sr = separate(full, audio_dir, models=models)
    proj.audio_vocals = str(voc)
    proj.audio_instrumental = str(inst)
    proj.sr = sr
    proj.save(proj_path)

    print(f"[separate] vocals={voc.name} instrumental={inst.name} sr={sr}")
    print(f"[separate] project -> {proj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
