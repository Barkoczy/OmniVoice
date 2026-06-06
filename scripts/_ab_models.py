"""A/B compare remote LM Studio models on hard EN->CS sentences (pick the best)."""

import json
import urllib.error
import urllib.request

HOST = "http://192.168.88.111:1234/v1/chat/completions"
MODELS = ["google/gemma-4-31b", "zai-org/glm-4.7-flash", "qwen3-vl-235b-a22b-thinking"]
SENTS = {
    1: "And the Czech Republic's Houska Castle is no exception.",
    3: "The Czech Republic has been called the castle capital of the world.",
    5: ("The most famous is Prague Castle, the centerpiece of the country's "
        "bustling capital and the largest castle complex in the world."),
}
SYS = ("You are a professional English-to-Czech translator. Translate into fluent, "
       "grammatically flawless Czech with correct case (pády) and agreement. "
       "Output ONLY the Czech translation, nothing else.")


def ask(model, text):
    body = {"model": model,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": text}],
            "temperature": 0.2, "max_tokens": 500}
    data = json.dumps(body).encode()
    req = urllib.request.Request(HOST, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            m = json.loads(r.read())["choices"][0]["message"]
        return (m.get("content") or m.get("reasoning_content") or "").strip().replace("\n", " ")
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


for model in MODELS:
    print(f"\n#### {model}", flush=True)
    for i, s in SENTS.items():
        print(f"[{i}] {ask(model, s)}", flush=True)
