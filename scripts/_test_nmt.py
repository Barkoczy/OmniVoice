"""Compare offline open-source EN->CS NMT models on hard declension sentences."""

import sys
import warnings

warnings.filterwarnings("ignore")
from transformers import pipeline  # noqa: E402

SENTS = [
    "Although the country's only about the size of South Carolina, it's got more than 2,000 of them.",
    "The Czech Republic has been called the castle capital of the world.",
    "And the Czech Republic's Houska Castle is no exception.",
    "The most famous is Prague Castle, the centerpiece of the country's bustling capital.",
]


def run(name, **kw):
    try:
        p = pipeline("translation", device=0, **kw)
        print(f"\n=== {name} ===", flush=True)
        for s in SENTS:
            print("  " + p(s, max_length=256)[0]["translation_text"], flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"\n=== {name} ERROR: {e!r}", flush=True)


run("Opus-MT en-cs (CC-BY)", model="Helsinki-NLP/opus-mt-en-cs")
run("NLLB-200-distilled-600M (CC-BY-NC)", model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn", tgt_lang="ces_Latn")
