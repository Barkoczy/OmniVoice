"""Transcribe a reference voice once and set it for all speakers in a project.

Usage (analysis venv):  python scripts/_set_narrator.py <ref.wav> <workdir>
Avoids OmniVoice re-running Whisper on the reference for every segment.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import whisperx  # noqa: E402
from dubbing.schema import DubProject  # noqa: E402

wav, workdir = sys.argv[1], sys.argv[2]
model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="cs")
result = model.transcribe(whisperx.load_audio(wav), batch_size=8)
text = " ".join(s["text"].strip() for s in result.get("segments", [])).strip()
print("REF TEXT:", text)

proj = DubProject.load(Path(workdir) / "project.json")
for spk in proj.speakers.values():
    spk.ref_voice = wav
    spk.ref_text = text
proj.save(Path(workdir) / "project.json")
print("speakers set:", list(proj.speakers))
