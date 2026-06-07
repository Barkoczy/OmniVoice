"""Phase 1b+1c: ASR (WhisperX large-v3) + diarization (pyannote) + gender.

Runs in the analysis venv. Transcribes the vocals stem with word-level
timestamps, assigns speakers via pyannote diarization, classifies each
speaker's gender, and writes the result into project.json plus a source SRT.

Diarization needs a Hugging Face token and one-time acceptance of the
pyannote/speaker-diarization-3.1 user conditions on huggingface.co.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dubbing.config import (ASR_SR, DIARIZATION_MODEL, GENDER_MODEL,  # noqa: E402
                            WHISPER_MODEL)
from dubbing.schema import DubProject, Segment, Speaker  # noqa: E402


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    tok_file = Path.home() / ".cache" / "huggingface" / "token"
    if tok_file.exists():
        return tok_file.read_text(encoding="utf-8").strip() or None
    return None


def _fmt_ts(t: float) -> str:
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60); t -= m * 60
    s = int(t); ms = int(round((t - s) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[Segment], path: Path, use_target: bool = False) -> Path:
    lines = []
    for i, seg in enumerate(segments, 1):
        text = (seg.text_tgt or seg.text_src) if use_target else seg.text_src
        lines.append(f"{i}\n{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n{text}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def transcribe_align_diarize(vocals: str, language: str | None, device: str = "cuda",
                             min_speakers=None, max_speakers=None) -> dict:
    import whisperx

    audio = whisperx.load_audio(vocals)
    model = whisperx.load_model(WHISPER_MODEL, device, compute_type="float16",
                                language=language)
    result = model.transcribe(audio, batch_size=16)
    lang = result.get("language", language) or "en"

    # Word-level alignment
    try:
        model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                return_char_alignments=False)
    except Exception as e:  # noqa: BLE001
        print(f"[transcribe] alignment skipped ({lang}): {e}")

    # Diarization (+ assign speakers to words/segments)
    try:
        try:
            from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        except ImportError:
            from whisperx import DiarizationPipeline, assign_word_speakers  # type: ignore
        diar = DiarizationPipeline(model_name=DIARIZATION_MODEL, token=hf_token(),
                                   device=device)
        diar_segments = diar(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = assign_word_speakers(diar_segments, result)
    except Exception as e:  # noqa: BLE001
        print(f"[transcribe] diarization skipped: {e}")

    result["language"] = lang
    return result


def classify_genders(vocals: str, segments: list[Segment], device: str = "cuda") -> dict:
    """Return {speaker_id: 'male'|'female'} using a wav2vec2 gender classifier.

    Runs the model manually (feature extractor + forward) to avoid the
    transformers audio pipeline's hard dependency on torchcodec.
    """
    try:
        import torch
        import torchaudio
        from transformers import (AutoFeatureExtractor,
                                  AutoModelForAudioClassification)
    except Exception as e:  # noqa: BLE001
        print(f"[gender] unavailable: {e}")
        return {}

    data, sr = sf.read(vocals)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)

    by_spk: dict[str, list[np.ndarray]] = {}
    for seg in segments:
        s, e = int(seg.start * sr), int(seg.end * sr)
        if e > s:
            by_spk.setdefault(seg.speaker, []).append(data[s:e])

    fe = AutoFeatureExtractor.from_pretrained(GENDER_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(GENDER_MODEL).to(device).eval()
    id2label = model.config.id2label

    genders = {}
    for spk, chunks in by_spk.items():
        clip = np.concatenate(chunks)[: sr * 8]
        if sr != 16000:
            clip = torchaudio.functional.resample(torch.from_numpy(clip), sr, 16000).numpy()
        inputs = fe(clip, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        label = str(id2label[int(logits.argmax(-1))]).lower()
        genders[spk] = "female" if "female" in label else "male"
    return genders


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WhisperX ASR + diarization + gender.")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--language", default=None, help="Source language (None=auto).")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args(argv)

    workdir = Path(args.workdir)
    proj = DubProject.load(workdir / "project.json")
    vocals = proj.audio_vocals
    if not vocals or not Path(vocals).exists():
        print("[transcribe] no vocals stem; run separate.py first", file=sys.stderr)
        return 2

    result = transcribe_align_diarize(
        vocals, args.language, min_speakers=args.min_speakers,
        max_speakers=args.max_speakers)
    proj.source_language = result.get("language")

    segments = []
    for i, seg in enumerate(result.get("segments", [])):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append(Segment(
            id=len(segments), start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)), text_src=text,
            speaker=seg.get("speaker", "SPEAKER_00")))
    proj.segments = segments

    genders = classify_genders(vocals, segments)
    for seg in segments:
        seg.gender = genders.get(seg.speaker, "unknown")
    proj.speakers = {
        spk: Speaker(id=spk, gender=genders.get(spk, "unknown"))
        for spk in sorted({s.speaker for s in segments})
    }

    proj.save(workdir / "project.json")
    srt = write_srt(segments, workdir / "source.srt")
    print(f"[transcribe] lang={proj.source_language} segments={len(segments)} "
          f"speakers={list(proj.speakers)} -> {srt.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
