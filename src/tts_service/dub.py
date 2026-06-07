"""SRT-based dubbing pipeline on top of :class:`tts_service.tts.CzechTTS`.

Given a subtitle file (.srt) and a reference voice, this synthesizes each
subtitle line as Czech speech, fits it into its original time slot, assembles a
single aligned audio track, and (optionally) muxes it back onto a video with
ffmpeg.

The key trick for lip/timing sync is OmniVoice's ``duration`` parameter: each
segment is rendered to exactly the length of its subtitle slot, with trailing
silence trimming disabled so the timing matches.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .tts import CzechTTS, SAMPLE_RATE

_TS = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_RE_CUE_TIME = re.compile(_TS + r"\s*-->\s*" + _TS)
_MIN_SLOT = 0.30  # seconds; OmniVoice is unreliable below this even with a reference


@dataclass
class Cue:
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str

    @property
    def slot(self) -> float:
        return self.end - self.start


def _ts_to_seconds(h, m, s, ms) -> float:
    ms = (ms + "000")[:3]  # normalize to milliseconds
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: str | Path) -> List[Cue]:
    """Parse an .srt file into a list of cues. Tolerant of blank lines / CRLF."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues: List[Cue] = []
    for block in blocks:
        lines = [ln.rstrip("\r") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        time_idx = 0
        if not _RE_CUE_TIME.search(lines[0]) and len(lines) > 1:
            time_idx = 1  # first line is the numeric index
        m = _RE_CUE_TIME.search(lines[time_idx]) if len(lines) > time_idx else None
        if not m:
            continue
        start = _ts_to_seconds(*m.group(1, 2, 3, 4))
        end = _ts_to_seconds(*m.group(5, 6, 7, 8))
        text = " ".join(lines[time_idx + 1:]).strip()
        if not text:
            continue
        cues.append(Cue(index=len(cues) + 1, start=start, end=end, text=text))
    return cues


def _ffprobe_duration(video: str | Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def assemble_track(clips: List[tuple], total_seconds: float) -> np.ndarray:
    """Place each (start_seconds, audio) clip on a single zeroed mono track."""
    total_samples = int(round(total_seconds * SAMPLE_RATE)) + 1
    track = np.zeros(total_samples, dtype=np.float32)
    for start, audio in clips:
        a = np.asarray(audio, dtype=np.float32)
        s = int(round(start * SAMPLE_RATE))
        e = min(s + len(a), total_samples)
        if s >= total_samples or e <= s:
            continue
        track[s:e] += a[: e - s]  # additive overlay (clips normally do not overlap)
    np.clip(track, -1.0, 1.0, out=track)
    return track


def mux_video(
    video: str | Path,
    dub_wav: str | Path,
    out_path: str | Path,
    keep_original: bool = False,
    original_volume: float = 0.25,
) -> None:
    """Mux the dubbed audio onto the video. Replaces audio unless keep_original."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH; cannot mux video.")
    video, dub_wav, out_path = str(video), str(dub_wav), str(out_path)
    if keep_original:
        cmd = [
            "ffmpeg", "-y", "-i", video, "-i", dub_wav,
            "-filter_complex",
            f"[0:a]volume={original_volume}[bg];[bg][1:a]amix=inputs=2:"
            "duration=longest:dropout_transition=0[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", video, "-i", dub_wav,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def dub_srt(
    srt_path: str | Path,
    ref_audio: Optional[str],
    out_audio: str | Path,
    *,
    tts: Optional[CzechTTS] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    language: str = "cs",
    fit: str = "duration",
    normalize: bool = True,
    video: Optional[str | Path] = None,
    out_video: Optional[str | Path] = None,
    keep_original: bool = False,
    original_volume: float = 0.25,
    num_step: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Dub a video/audio track from an .srt file.

    fit:
        ``"duration"`` (default) fits each line to its exact subtitle slot
        (best timing sync). ``"natural"`` renders at natural pace and only
        anchors the start time (may overrun a slot).

    Returns a summary dict.
    """
    if ref_audio is None and instruct is None:
        raise ValueError(
            "Provide ref_audio (a 3-10s Czech voice sample to clone) or instruct. "
            "Without one, every line would get a different random voice."
        )
    if fit not in ("duration", "natural"):
        raise ValueError("fit must be 'duration' or 'natural'")

    tts = tts or CzechTTS()
    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError(f"No subtitle cues found in {srt_path}")

    import soundfile as sf

    clips: List[tuple] = []
    skipped: List[int] = []
    for cue in cues:
        target = None
        exact = False
        if fit == "duration":
            target = max(cue.slot, _MIN_SLOT)
            exact = True
            if cue.slot < _MIN_SLOT and verbose:
                print(f"  [warn] cue {cue.index}: slot {cue.slot:.2f}s < {_MIN_SLOT}s, padded")
        audio = tts.synthesize(
            cue.text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            language=language,
            duration=target,
            normalize=normalize,
            exact_duration=exact,
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
        if fit == "duration":
            # Clip to the exact slot so a segment never overruns into the next.
            max_len = int(round(cue.slot * SAMPLE_RATE))
            if len(audio) > max_len:
                audio = audio[:max_len]
        clips.append((cue.start, audio))
        if verbose:
            print(f"  cue {cue.index}/{len(cues)} @ {cue.start:6.2f}s "
                  f"slot={cue.slot:4.2f}s out={len(audio)/SAMPLE_RATE:4.2f}s", flush=True)

    video_dur = _ffprobe_duration(video) if video else None
    total = max([c.end for c in cues] + ([video_dur] if video_dur else []))
    track = assemble_track(clips, total)

    out_audio = Path(out_audio)
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_audio), track, SAMPLE_RATE)

    result = {
        "cues": len(cues),
        "skipped": skipped,
        "total_seconds": round(total, 2),
        "audio": str(out_audio),
        "video": None,
    }

    if video:
        out_video = Path(out_video) if out_video else out_audio.with_suffix(".mp4")
        mux_video(video, out_audio, out_video, keep_original, original_volume)
        result["video"] = str(out_video)

    return result
