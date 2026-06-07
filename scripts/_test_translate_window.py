"""Hard-data check of the fixed translate_window: thinking ON, JSON read from
content, generous token budget (MAX_TOKENS) + salvage. Runs a real window of film
segments against the configured backend (.env) and reports clean recovery.

Usage: _test_translate_window.py [start_index] [workdir]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dubbing import translate  # noqa: E402  (backend comes from .env via config)
from dubbing.schema import DubProject  # noqa: E402

start = int(sys.argv[1]) if len(sys.argv) > 1 else 200
workdir = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "output" / "The-House-of-Rothschild-1934"

proj = DubProject.load(workdir / "project.json")
win = proj.segments[start:start + translate.WINDOW]
src = proj.source_language or "en"
print(f"backend={translate.BASE_URL} model={translate.MODEL} "
      f"window={len(win)} max_tokens={translate.MAX_TOKENS}")

mapping = translate.translate_window(win, [], src, "cs")
for s in win:
    tag = "OK  " if s.id in mapping else "MISS"
    print(f"[{tag}] {s.id}: {s.text_src[:50]!r} -> {(mapping.get(s.id) or '')[:60]!r}")
print(f"\nrecovered {len(mapping)}/{len(win)} lines")
