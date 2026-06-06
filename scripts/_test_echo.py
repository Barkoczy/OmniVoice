"""Quick test: does a short single-utterance original ref still echo into output?"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dubbing.schema import DubProject  # noqa: E402
from dubbing.synth import extract_speaker_ref  # noqa: E402
from tts_service.tts import SAMPLE_RATE, CzechTTS  # noqa: E402

wd = ROOT / "output" / "the-creepy-origin-of-the-abandoned-houska-castle-l-legendary-locations-en"
proj = DubProject.load(wd / "project.json")
rv, rt = extract_speaker_ref(proj, "SPEAKER_00", wd / "voices")
ref_dur = len(sf.read(rv)[0]) / sf.info(rv).samplerate
print(f"REF: {Path(rv).name} ({ref_dur:.1f}s) | ref_text: {rt}")

tts = CzechTTS(num_step=64)
segs = [s for s in proj.segments if s.speaker == "SPEAKER_00"][:3]
clips = []
for s in segs:
    dur = s.slot if s.slot >= 1.0 else None
    a = tts.synthesize(s.text_tts, ref_audio=rv, ref_text=rt,
                       language="cs", duration=dur, normalize=False)
    print(f"  seg {s.id}: {len(a)/SAMPLE_RATE:.1f}s  cz='{s.text_tts[:50]}'")
    clips.append(a)
sf.write(str(wd / "_echo_test.wav"), np.concatenate(clips), SAMPLE_RATE)
print("wrote _echo_test.wav")
