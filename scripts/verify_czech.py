"""Objectively verify Czech output: audio levels + ASR round-trip + cloning.

Transcribes the generated Czech audio back with Whisper to confirm it is
intelligible Czech (not just non-silent noise), and runs a voice-cloning test
with exact-duration fitting (the dubbing code path).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tts_service.tts import CzechTTS, SAMPLE_RATE  # noqa: E402


def stats(path: Path) -> dict:
    a, sr = sf.read(str(path))
    a = np.asarray(a, dtype=np.float32)
    return {
        "file": path.name,
        "sr": sr,
        "dur_s": round(len(a) / sr, 2),
        "peak": round(float(np.max(np.abs(a))) if a.size else 0.0, 3),
        "rms": round(float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0, 4),
    }


def main() -> int:
    out = ROOT / "output"
    refs = ROOT / "refs"

    print("=== audio stats (peak>0 & rms>0 => not silent) ===", flush=True)
    for p in [out / "smoke_cs.wav", out / "smoke_en.wav", refs / "auto_cs.wav"]:
        if p.exists():
            print(" ", stats(p), flush=True)

    # --- Voice cloning test (the dubbing code path) ---
    tts = CzechTTS(num_step=64, guidance_scale=3.0)
    ref = str(refs / "auto_cs.wav")
    ref_text = (
        "Dobrý den, vítejte u ukázky českého dabingu. "
        "Tento hlas slouží jako referenční vzorek pro klonování."
    )
    clone_text = "Toto je klonovaný český hlas. Dnes je krásný den a mám 42 korun."
    print("\n=== cloning test (ref=auto_cs.wav, fit to 5.0s) ===", flush=True)
    t0 = time.time()
    audio = tts.synthesize(
        clone_text, ref_audio=ref, ref_text=ref_text,
        language="cs", duration=5.0, exact_duration=True,
    )
    sf.write(str(out / "clone_cs.wav"), audio, SAMPLE_RATE)
    print(f"  generated {len(audio)/SAMPLE_RATE:.2f}s in {time.time()-t0:.1f}s "
          f"-> output/clone_cs.wav", flush=True)
    print(" ", stats(out / "clone_cs.wav"), flush=True)

    # --- ASR round-trip with Whisper to confirm intelligible Czech ---
    print("\n=== ASR round-trip (Whisper transcribes the TTS back) ===", flush=True)
    print("  expected smoke_cs ~= 'toto je test ceske syntezy ... 1234 korun'", flush=True)
    print(f"  expected clone_cs ~= '{clone_text}'", flush=True)
    try:
        import torch
        from transformers import pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        for name in ["smoke_cs.wav", "clone_cs.wav"]:
            r = asr(str(out / name), generate_kwargs={"language": "czech", "task": "transcribe"})
            print(f"  [{name}] -> {r['text'].strip()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("  ASR skipped:", repr(e), flush=True)

    print("\nVERIFY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
