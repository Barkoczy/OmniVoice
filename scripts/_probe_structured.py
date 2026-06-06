"""Hard-data probe: does the configured LLM backend support reliable structured
output (response_format json_schema), and is it a reasoning model whose thinking
stream gets constrained (bug #1773)? Prints raw evidence, no guessing.

Backend comes from .env / config (LLM_BASE_URL/MODEL/API_KEY); override via argv.
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dubbing.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL  # noqa: E402

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else LLM_BASE_URL).rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else LLM_MODEL
API_KEY = sys.argv[3] if len(sys.argv) > 3 else LLM_API_KEY
ENDPOINT = BASE_URL + "/chat/completions"

PAYLOAD = [{"id": 1, "text": "South Carolina is a US state."},
           {"id": 2, "text": "The castle was built to keep enemies inside."}]
USER = ("Translate each line's text to Czech. Return JSON "
        '{"translations":[{"id":<int>,"text":<str>}]}.\n\n'
        + json.dumps(PAYLOAD, ensure_ascii=False))

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"},
                                       "text": {"type": "string"}},
                        "required": ["id", "text"],
                    },
                }
            },
            "required": ["translations"],
        },
    },
}


def call(label, use_schema):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": USER}],
            "temperature": 0.2, "max_tokens": 2048}
    if use_schema:
        body["response_format"] = SCHEMA
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, headers=headers)
    print(f"\n===== {label} (response_format={'json_schema' if use_schema else 'none'}) =====")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
    except Exception as e:  # noqa: BLE001
        print(f"REQUEST ERROR: {type(e).__name__}: {e}")
        return
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    print(f"content_len={len(content)}  reasoning_content_len={len(reasoning)}")
    print(f"content[:400]={content[:400]!r}")
    if reasoning:
        print(f"reasoning[:200]={reasoning[:200]!r}")
    try:
        obj = json.loads(content)
        print(f"PARSED OK: {len(obj.get('translations', []))} translations -> {obj}")
    except Exception as e:  # noqa: BLE001
        print(f"PARSE FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print(f"endpoint={ENDPOINT} model={MODEL}")
    call("A plain", use_schema=False)
    call("B json_schema", use_schema=True)
