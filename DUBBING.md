# Video dubbing pipeline (any language → Czech)

End-to-end local dubbing built on OmniVoice. Upload a video, get it dubbed into
Czech (or another language) with **separate music preserved**, **one unique
gender-matched voice per speaker**, and **context- and gender-aware translation**.

## Pipeline

```
video ─▶ separate ─▶ transcribe ─▶ translate ─▶ synth ─▶ assemble ─▶ dubbed video
         (Demucs/      (WhisperX     (LM Studio    (OmniVoice  (mix dub +
          Roformer)     large-v3 +    gemma-4-31b)  per-speaker) original music,
                        pyannote +                               ffmpeg mux)
                        gender)
```

Each phase runs as **its own subprocess in its own virtualenv**, so the OS frees
all GPU memory between phases. That is what lets the ~19 GB translation LLM and
the analysis/TTS models share a single 24 GB GPU — they are never resident at the
same time. Verified: after the translation phase the LLM unloads and VRAM returns
to ~22.5 GB free before TTS starts.

| Phase | Env | Models |
|---|---|---|
| separate | `.venv-analysis` | Demucs htdemucs_ft (+ BS-Roformer ensemble) |
| transcribe | `.venv-analysis` | WhisperX large-v3, pyannote 3.1, wav2vec2 gender |
| translate | `.venv` | LM Studio `gemma-4-31b-qat` (load → translate → unload) |
| synth | `.venv` | OmniVoice (per-speaker Czech voices, fit to slots) |
| assemble | `.venv` | ffmpeg (mix dubbed vocals + original instrumental) |

## Prerequisites

- **LM Studio** running with `google/gemma-4-31b-qat` downloaded (already present).
  The translate phase starts the server and loads/unloads the model itself.
- **Hugging Face token** in `~/.cache/huggingface/token` (already present) and a
  one-time acceptance of the `pyannote/speaker-diarization-3.1` user conditions.
- ffmpeg on PATH (already present).

## Usage

```powershell
# Full pipeline (English video -> Czech dub)
.\.venv\Scripts\python.exe scripts\dub_pipeline.py --video movie.mp4 --source en --target cs

# Auto-detect source, constrain speaker count, highest-quality separation
.\.venv\Scripts\python.exe scripts\dub_pipeline.py --video clip.mkv --target cs `
    --separation ensemble --min-speakers 2 --max-speakers 4

# Run a subset of phases (e.g. re-synthesize after editing the translation)
.\.venv\Scripts\python.exe scripts\dub_pipeline.py --video movie.mp4 --steps synth,assemble
```

Supported containers: mp4, mkv, avi, mov, webm, m4v, flv, ts. Output and the
intermediate `project.json` (the shared data contract) land in `output/<name>/`.

## Per-phase (advanced)

```powershell
.\.venv-analysis\Scripts\python.exe src\dubbing\separate.py   --input movie.mp4 --workdir output\movie --models ensemble
.\.venv-analysis\Scripts\python.exe src\dubbing\transcribe.py --workdir output\movie --language en
.\.venv\Scripts\python.exe          src\dubbing\translate.py  --workdir output\movie --target cs
.\.venv\Scripts\python.exe          src\dubbing\synth.py      --workdir output\movie
.\.venv\Scripts\python.exe          src\dubbing\assemble.py   --workdir output\movie
```

## Voices per speaker

Each diarized speaker gets a distinct, gender-matched Czech voice. By default a
unique seed voice is generated per speaker (gender + a rotating pitch). To use a
real cloned voice instead, set `speakers.<id>.ref_voice` (and `ref_text`) in
`project.json` to a 3–10 s wav before the `synth` phase.

## Translation quality

The translator is prompted to preserve each speaker's grammatical gender (Czech
`řekl` vs `řekla`), keep domain terminology and anglicisms natural, hold context
across lines, and match register. For a Czech target the result is additionally
run through the Czech TTS normalizer (numbers → words, abbreviations, units).
