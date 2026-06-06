"""Central configuration: interpreter paths, model names, service endpoints."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency): KEY=VALUE lines, '#' comments.
    Real environment variables win (setdefault), so a shell export still overrides
    the file."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Per-phase Python interpreters (subprocess-per-phase keeps VRAM isolated).
VENV_TTS = ROOT / ".venv" / "Scripts" / "python.exe"            # OmniVoice
VENV_ANALYSIS = ROOT / ".venv-analysis" / "Scripts" / "python.exe"  # WhisperX/Demucs/diarization
VENV_ROFORMER = ROOT / ".venv-roformer" / "Scripts" / "python.exe"  # UVR BS-Roformer (audio-separator)
VENV_PARAKEET = ROOT / ".venv-parakeet" / "Scripts" / "python.exe"  # NVIDIA Parakeet (NeMo)

# Models
DEMUCS_MODEL = "htdemucs_ft"
# UVR BS-Roformer checkpoint name as exposed by the `audio-separator` package.
ROFORMER_MODEL = os.environ.get(
    "ROFORMER_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
WHISPER_MODEL = "large-v3"
PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

# LLM translation backend — provider-agnostic, OpenAI-compatible Chat Completions.
# Works with LM Studio, OpenAI, Google (OpenAI-compat), Mistral, Groq, Ollama, etc.
# Point LLM_BASE_URL at the provider's base (ending in /v1), then set model + key.
# Configure per machine in .env (see .env.example); nothing here is tool-specific.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-31b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# Approx VRAM footprint (GB) of a LOCAL LLM, used only to size context when a local
# LM Studio is VRAM-managed; irrelevant for remote/cloud backends.
LLM_WEIGHTS_GB = float(os.environ.get("LLM_WEIGHTS_GB", "19.0"))

# Audio working format
SEP_SR = 44100          # separation / instrumental stem sample rate
ASR_SR = 16000          # ASR input sample rate
TTS_SR = 24000          # OmniVoice output sample rate
