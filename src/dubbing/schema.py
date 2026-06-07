"""Shared data contract between dubbing phases (serialized to project.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text_src: str = ""          # original transcription (primary ASR: WhisperX)
    text_asr2: str = ""         # secondary ASR (Parakeet) for the same span
    asr_agreement: float = 1.0  # 0..1 similarity between the two ASR outputs
    text_tgt: str = ""          # primary translation (LLM, context-aware)
    text_nmt: str = ""          # reference translation (offline NMT) for grammar refine
    text_tts: str = ""          # final translation normalized for TTS
    speaker: str = "SPEAKER_00"
    gender: str = "unknown"     # male | female | unknown

    @property
    def slot(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Speaker:
    id: str
    gender: str = "unknown"
    ref_voice: Optional[str] = None   # path to the cloned reference wav
    ref_text: Optional[str] = None


@dataclass
class DubProject:
    source_video: Optional[str] = None
    workdir: str = "."
    source_language: Optional[str] = None   # None = auto-detect
    target_language: str = "cs"
    sr: int = 44100
    audio_full: Optional[str] = None
    audio_vocals: Optional[str] = None
    audio_instrumental: Optional[str] = None
    speakers: Dict[str, Speaker] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)

    # --- serialization ---
    def to_dict(self) -> dict:
        d = asdict(self)
        d["speakers"] = {k: asdict(v) for k, v in self.speakers.items()}
        d["segments"] = [asdict(s) for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DubProject":
        speakers = {k: Speaker(**v) for k, v in (d.get("speakers") or {}).items()}
        segments = [Segment(**s) for s in (d.get("segments") or [])]
        known = {f for f in cls.__dataclass_fields__ if f not in ("speakers", "segments")}
        base = {k: v for k, v in d.items() if k in known}
        return cls(speakers=speakers, segments=segments, **base)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else Path(self.workdir) / "project.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DubProject":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
