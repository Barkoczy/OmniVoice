"""Smoke test: download the model (first run) and synthesize EN + CS samples.

Run:  .venv\\Scripts\\python.exe scripts\\smoke_test.py
Outputs land in ./output and a reusable Czech demo voice in ./refs.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tts_service.tts import CzechTTS, SAMPLE_RATE  # noqa: E402


def _dur(audio) -> float:
    return len(audio) / SAMPLE_RATE


def main() -> int:
    out = ROOT / "output"
    refs = ROOT / "refs"
    out.mkdir(exist_ok=True)
    refs.mkdir(exist_ok=True)

    print("Loading OmniVoice (first run downloads the weights)...", flush=True)
    t0 = time.time()
    tts = CzechTTS(num_step=48)
    _ = tts.model  # trigger load/download now
    print(f"  model ready in {time.time() - t0:.1f}s", flush=True)

    import soundfile as sf

    # 1) English auto-voice sanity check
    t1 = time.time()
    en = tts.synthesize(
        "Hello, this is OmniVoice running locally on an RTX 4090.",
        language="en",
        normalize=False,
    )
    sf.write(str(out / "smoke_en.wav"), en, SAMPLE_RATE)
    print(f"[EN] {_dur(en):.1f}s audio in {time.time() - t1:.1f}s -> output/smoke_en.wav", flush=True)

    # 2) Czech auto-voice + normalization (numbers must be spoken correctly)
    cz_text = (
        "Ahoj, toto je test české syntézy řeči pomocí modelu OmniVoice. "
        "Mám 3 jablka a celková cena je 1 234 korun."
    )
    t2 = time.time()
    cz = tts.synthesize(cz_text, language="cs")
    sf.write(str(out / "smoke_cs.wav"), cz, SAMPLE_RATE)
    print(f"[CS] {_dur(cz):.1f}s audio in {time.time() - t2:.1f}s -> output/smoke_cs.wav", flush=True)

    # 3) Longer Czech clip saved as a reusable demo reference voice for cloning
    ref_text = (
        "Dobrý den, vítejte u ukázky českého dabingu. "
        "Tento hlas slouží jako referenční vzorek pro klonování."
    )
    cz_ref = tts.synthesize(ref_text, language="cs", normalize=False)
    sf.write(str(refs / "auto_cs.wav"), cz_ref, SAMPLE_RATE)
    print(f"[REF] {_dur(cz_ref):.1f}s demo reference -> refs/auto_cs.wav", flush=True)

    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak VRAM allocated: {peak:.2f} GB", flush=True)
    except Exception:
        pass

    print("SMOKE TEST OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
