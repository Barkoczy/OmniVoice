"""End-to-end video dubbing pipeline on top of OmniVoice.

Pipeline phases (each heavy phase runs as its own subprocess so VRAM is fully
released on exit):

1. separate  - extract audio, split vocals (speech) vs instrumental (music/SFX)
2. asr       - WhisperX (large-v3) word-level transcription + Parakeet consensus
3. diarize   - pyannote speaker turns + per-speaker gender
4. translate - LM Studio (gemma-4-31b-qat), gender/context-aware, any -> any
5. synth     - per-speaker Czech voices via OmniVoice, fit to slots
6. assemble  - mix dubbed vocals + original instrumental, mux onto video

The shared data contract between phases is :class:`dubbing.schema.DubProject`,
serialized to ``project.json`` in the work directory.
"""

from .schema import DubProject, Segment, Speaker

__all__ = ["DubProject", "Segment", "Speaker"]
