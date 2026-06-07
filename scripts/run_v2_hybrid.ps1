# Run the full v2 HYBRID dub of a film FROM SCRATCH.
#
# Hybrid = offline Opus-MT grammar reference (text_nmt) + context-aware LLM translation
# (gemma) with NMT-aware grammar polish. The LLM backend is taken from .env
# (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY) -> no host/model flags needed here.
#
# Prerequisites (check before launching):
#   - The LLM backend in .env is reachable (Mac LM Studio up, model loaded).
#     Quick check:  .\.venv\Scripts\python.exe scripts\_probe_mac_ctx.py
#   - The local GPU is free (separation / WhisperX / OmniVoice run locally).
#   - The .venv / .venv-analysis / .venv-roformer envs exist.
#
# Steps deliberately EXCLUDE Parakeet consensus (it OOMs on feature-length audio and
# does not affect the dub output). Expect ~3-4 h end to end for an ~87 min film.

param(
    [string]$Video = "C:\Users\henri\Downloads\The-House-of-Rothschild-1934.mp4"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "[v2-hybrid] video : $Video"
Write-Host "[v2-hybrid] backend from .env (LLM_BASE_URL / LLM_MODEL)"

& .\.venv\Scripts\python.exe scripts\dub_pipeline.py `
    --video $Video `
    --source en --target cs `
    --translator hybrid `
    --steps separate,transcribe,translate,synth,assemble
