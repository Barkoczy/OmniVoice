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
from dubbing.config import ROOT, VENV_ANALYSIS, VENV_TTS  # noqa: E402

SRC = ROOT / "src" / "dubbing"

# phase -> (interpreter, script)
PHASES = {
    "separate": (VENV_ANALYSIS, "separate.py"),
    "transcribe": (VENV_ANALYSIS, "transcribe.py"),
    "translate": (VENV_TTS, "translate.py"),
    "synth": (VENV_TTS, "synth.py"),
    "assemble": (VENV_TTS, "assemble.py"),
}
DEFAULT_STEPS = ["separate", "transcribe", "translate", "synth", "assemble"]


def _run(py: Path, script: str, *args) -> None:
    cmd = [str(py), str(SRC / script), *map(str, args)]
    print(f"\n========== {script} ==========", flush=True)
    subprocess.run(cmd, check=True)


def run_pipeline(video: str, workdir: str, *, source=None, target="cs",
                 separation="ensemble", min_speakers=None, max_speakers=None,
                 steps=None) -> dict:
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
    if "translate" in steps:
        _run(VENV_TTS, "translate.py", "--workdir", workdir, "--target", target)
    if "synth" in steps:
        _run(VENV_TTS, "synth.py", "--workdir", workdir)
    if "assemble" in steps:
        _run(VENV_TTS, "assemble.py", "--workdir", workdir)

    return {"workdir": workdir}
