"""Central configuration: interpreter paths, model names, service endpoints."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Per-phase Python interpreters (subprocess-per-phase keeps VRAM isolated).
VENV_TTS = ROOT / ".venv" / "Scripts" / "python.exe"            # OmniVoice
VENV_ANALYSIS = ROOT / ".venv-analysis" / "Scripts" / "python.exe"  # ASR/separation/diarization

# Models
DEMUCS_MODEL = "htdemucs_ft"
# UVR BS-Roformer checkpoint name as exposed by the `audio-separator` package.
ROFORMER_MODEL = os.environ.get(
    "ROFORMER_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
WHISPER_MODEL = "large-v3"
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

# LM Studio (translation)
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "google/gemma-4-31b-qat")
LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://localhost:1234")
LMSTUDIO_API = LMSTUDIO_HOST + "/v1"

# Approx VRAM footprint of the LLM weights (GB) for budgeting (gemma-4-31b-qat).
LLM_WEIGHTS_GB = float(os.environ.get("LLM_WEIGHTS_GB", "19.0"))

# Audio working format
SEP_SR = 44100          # separation / instrumental stem sample rate
ASR_SR = 16000          # ASR input sample rate
TTS_SR = 24000          # OmniVoice output sample rate
