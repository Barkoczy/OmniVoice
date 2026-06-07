"""Read-only probe: which models the LLM host has loaded and with what context
length. Uses LM Studio's native /api/v0 endpoint (richer than /v1/models).

Host defaults to the configured backend (.env LLM_BASE_URL, with its /v1 stripped);
override via argv. Only meaningful for an LM Studio host.
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dubbing.config import LLM_BASE_URL  # noqa: E402

HOST = (sys.argv[1] if len(sys.argv) > 1 else LLM_BASE_URL).rsplit("/v1", 1)[0].rstrip("/")


def get(path):
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.loads(r.read())


print(f"host={HOST}")
try:
    data = get("/api/v0/models")
    rows = data.get("data", data) if isinstance(data, dict) else data
    for m in rows:
        if not isinstance(m, dict):
            print(m)
            continue
        keys = ("id", "type", "state", "max_context_length", "loaded_context_length",
                "quantization", "arch")
        print({k: m.get(k) for k in keys if k in m})
except Exception as e:  # noqa: BLE001
    print(f"/api/v0/models ERROR: {type(e).__name__}: {e}")
