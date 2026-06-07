"""Probe which chat params the LM Studio OpenAI endpoint accepts (debug 400s)."""

import json
import urllib.error
import urllib.request

API = "http://localhost:1234/v1/chat/completions"
MODEL = "google/gemma-4-31b-qat"


def test(name, extra):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": 'Reply with JSON {"ok": true}'}],
        "max_tokens": 50,
        **extra,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(f"{name}: OK {r.status}")
    except urllib.error.HTTPError as e:
        print(f"{name}: HTTP {e.code} -> {e.read().decode()[:300]}")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: ERR {e!r}")


test("json_object+repeat_penalty",
     {"response_format": {"type": "json_object"}, "frequency_penalty": 0.6,
      "presence_penalty": 0.3, "top_p": 0.9, "repeat_penalty": 1.3})
test("json_object+std_penalties",
     {"response_format": {"type": "json_object"}, "frequency_penalty": 0.6,
      "presence_penalty": 0.3, "top_p": 0.9})
test("json_object_only", {"response_format": {"type": "json_object"}})
test("json_schema+std_penalties",
     {"response_format": {"type": "json_schema", "json_schema": {
         "name": "x", "schema": {"type": "object",
                                 "properties": {"ok": {"type": "boolean"}},
                                 "required": ["ok"]}, "strict": True}},
      "frequency_penalty": 0.6, "presence_penalty": 0.3, "top_p": 0.9})
test("plain", {})
