"""Pick a clean ~4-7s single-sentence segment from a sample, set it as the
reference voice for all speakers (avoids music/credits/hallucinations)."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import whisperx  # noqa: E402

from dubbing.schema import DubProject  # noqa: E402

sample, workdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="cs")
segs = m.transcribe(whisperx.load_audio(sample), batch_size=8)["segments"]

_BAD = ("titulky", "johnyx", "amara", "překlad")  # common Whisper hallucinations


def good(s):
    d = s["end"] - s["start"]
    t = s["text"].strip()
    return 3.0 <= d <= 7.0 and len(t) > 15 and not any(b in t.lower() for b in _BAD)


cand = [s for s in segs if good(s)] or [s for s in segs if (s["end"] - s["start"]) >= 2.5]
if not cand:
    print("no usable segment found", file=sys.stderr)
    raise SystemExit(2)
best = max(cand, key=lambda s: len(s["text"]))

data, sr = sf.read(sample)
data = np.asarray(data, dtype=np.float32)
if data.ndim > 1:
    data = data.mean(axis=1)
clip_dur = min(best["end"] - best["start"], 6.0)  # keep it short to avoid echo
start = best["start"]
clip = data[int(start * sr):int((start + clip_dur) * sr)]
sf.write(out, clip, sr)
# Re-transcribe just the short clip so ref_text matches the audio exactly.
rt = m.transcribe(whisperx.load_audio(out), batch_size=8)["segments"]
text = " ".join(s["text"].strip() for s in rt).strip()
print(f"REF: {out} ({clip_dur:.1f}s) | text: {text}")

proj = DubProject.load(Path(workdir) / "project.json")
for spk in proj.speakers.values():
    spk.ref_voice = out
    spk.ref_text = text
proj.save(Path(workdir) / "project.json")
print("speakers set:", list(proj.speakers))
