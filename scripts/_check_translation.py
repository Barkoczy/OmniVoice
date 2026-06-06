"""Scan ALL translated segments for leaked-reasoning contamination + show samples."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dubbing.schema import DubProject  # noqa: E402
from dubbing.translate import _is_contaminated  # noqa: E402

workdir = sys.argv[1] if len(sys.argv) > 1 else "."
proj = DubProject.load(Path(workdir) / "project.json")

bad = [(s.id, s.text_tgt) for s in proj.segments
       if _is_contaminated(s.text_src, s.text_tgt)]
print(f"segments: {len(proj.segments)} | contaminated: {len(bad)}")
for i, t in bad:
    print(f"  [BAD {i}] {t[:140]}")

print("\n--- first 8 ---")
for s in proj.segments[:8]:
    print(f"[{s.id}] {s.text_tts}")
print("--- last 4 ---")
for s in proj.segments[-4:]:
    print(f"[{s.id}] {s.text_tts}")
