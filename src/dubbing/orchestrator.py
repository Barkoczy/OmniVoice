"""Sequential, VRAM-isolated orchestration of the dubbing phases.

Each phase runs as its own subprocess in the correct virtual environment, so the
operating system fully reclaims its GPU memory when the process exits. This is
what lets the ~19 GB translation LLM and the analysis/TTS models share a single
24 GB GPU without ever being resident at the same time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.config import (ROOT, VENV_ANALYSIS, VENV_PARAKEET,  # noqa: E402
                            VENV_TTS)

SRC = ROOT / "src" / "dubbing"

# phase -> (interpreter, script)
PHASES = {
    "separate": (VENV_ANALYSIS, "separate.py"),
    "transcribe": (VENV_ANALYSIS, "transcribe.py"),
    "consensus": (VENV_PARAKEET, "parakeet.py"),
    "translate": (VENV_TTS, "translate.py"),
    "synth": (VENV_TTS, "synth.py"),
    "assemble": (VENV_TTS, "assemble.py"),
}
DEFAULT_STEPS = ["separate", "transcribe", "consensus", "translate", "synth", "assemble"]


def _run(py: Path, script: str, *args) -> None:
    cmd = [str(py), str(SRC / script), *map(str, args)]
    print(f"\n========== {script} ==========", flush=True)
    subprocess.run(cmd, check=True)


def run_pipeline(video: str, workdir: str, *, source=None, target="cs",
                 separation="ensemble", min_speakers=None, max_speakers=None,
                 steps=None, narrator=None, voices=None, polish=True,
                 base_url=None, api_key=None, model=None, translator="nmt",
                 nmt_model=None) -> dict:
    steps = steps or DEFAULT_STEPS

    if "separate" in steps:
        _run(VENV_ANALYSIS, "separate.py", "--input", video,
             "--workdir", workdir, "--models", separation)
    if "transcribe" in steps:
        args = ["--workdir", workdir]
        if source:
            args += ["--language", source]
        if min_speakers:
            args += ["--min-speakers", min_speakers]
        if max_speakers:
            args += ["--max-speakers", max_speakers]
        _run(VENV_ANALYSIS, "transcribe.py", *args)
    if "consensus" in steps:
        try:
            _run(VENV_PARAKEET, "parakeet.py", "--workdir", workdir)
        except Exception as e:  # noqa: BLE001 - secondary ASR is optional
            print(f"[orchestrator] consensus (Parakeet) skipped: {e}")
    if "translate" in steps:
        # nmt: offline NMT only. llm: LLM only. hybrid: NMT grammar reference + LLM
        # (context) with NMT-aware grammar polish.
        if translator in ("nmt", "hybrid"):
            nmt_args = ["--workdir", workdir, "--target", target,
                        "--field", "text_nmt" if translator == "hybrid" else "text_tgt"]
            if source:
                nmt_args += ["--source", source]
            if nmt_model:
                nmt_args += ["--model", nmt_model]
            _run(VENV_ANALYSIS, "translate_nmt.py", *nmt_args)
        if translator in ("llm", "hybrid"):
            tr_args = ["--workdir", workdir, "--target", target]
            if not polish:
                tr_args.append("--no-polish")
            if base_url:
                tr_args += ["--base-url", base_url]
            if api_key:
                tr_args += ["--api-key", api_key]
            if model:
                tr_args += ["--model", model]
            _run(VENV_TTS, "translate.py", *tr_args)
    if "synth" in steps:
        synth_args = ["--workdir", workdir]
        if narrator:
            synth_args += ["--narrator", narrator]
        for v in (voices or []):
            synth_args += ["--voice", v]
        _run(VENV_TTS, "synth.py", *synth_args)
    if "assemble" in steps:
        _run(VENV_TTS, "assemble.py", "--workdir", workdir)

    return {"workdir": workdir}
