"""Thin, Czech-friendly wrapper around the OmniVoice model.

The wrapper lazily loads the model (so importing the module is cheap), applies
Czech text normalization by default, and exposes sensible defaults for dubbing
work. The underlying model and its parameters are documented at
https://github.com/k2-fsa/OmniVoice .
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .cz_normalize import normalize_text

DEFAULT_MODEL = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000  # OmniVoice always outputs 24 kHz mono

_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


class CzechTTS:
    """Convenience wrapper producing 24 kHz mono speech with Czech defaults.

    Parameters
    ----------
    model_name:
        HuggingFace id or local path of the OmniVoice checkpoint.
    device:
        Torch device map, e.g. ``"cuda:0"``, ``"cpu"``, ``"mps"``, ``"xpu"``.
    dtype:
        ``"float16"`` (default, fastest on the RTX 4090), ``"bfloat16"`` or
        ``"float32"``.
    num_step:
        Diffusion steps. 32 = model default, 48 = good quality/speed balance,
        64 = best quality (the value recommended in the OmniVoice walkthrough).
    guidance_scale:
        Classifier-free guidance. 2.0 = model default. Raise to 3.0-4.0 for
        stronger adherence to the reference voice when cloning.
    language:
        OmniVoice language id; ``"cs"`` for Czech. ``None`` = auto-detect.
    normalize:
        Apply Czech text normalization (numbers, units, abbreviations) by
        default. Can be overridden per call.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        dtype: str = "float16",
        num_step: int = 48,
        guidance_scale: float = 2.0,
        language: Optional[str] = "cs",
        normalize: bool = True,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}, got {dtype!r}")
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.num_step = num_step
        self.guidance_scale = guidance_scale
        self.language = language
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        """Load the OmniVoice model on first use (cached afterwards)."""
        if self._model is None:
            import torch
            from omnivoice import OmniVoice

            torch_dtype = getattr(torch, self.dtype)
            self._model = OmniVoice.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=torch_dtype,
            )
        return self._model

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: Optional[str] = "__default__",
        duration: Optional[float] = None,
        speed: Optional[float] = None,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        normalize: Optional[bool] = None,
        exact_duration: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize a single string and return a float32 numpy array (24 kHz).

        ``exact_duration=True`` disables trailing-silence trimming so the output
        length matches ``duration`` exactly (useful for fitting dubbing slots).
        """
        if normalize is None:
            normalize = self.normalize
        if normalize:
            text = normalize_text(text)
        if language == "__default__":
            language = self.language

        gen_kwargs = dict(kwargs)
        gen_kwargs.setdefault("num_step", num_step if num_step is not None else self.num_step)
        gen_kwargs.setdefault(
            "guidance_scale",
            guidance_scale if guidance_scale is not None else self.guidance_scale,
        )
        if exact_duration:
            gen_kwargs["postprocess_output"] = False

        audio = self.model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            language=language,
            duration=duration,
            speed=speed,
            **gen_kwargs,
        )
        return audio[0]

    def to_file(self, text: str, out_path: Union[str, Path], **kwargs) -> Path:
        """Synthesize ``text`` and write it to ``out_path`` as a WAV file."""
        import soundfile as sf

        audio = self.synthesize(text, **kwargs)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, SAMPLE_RATE)
        return out_path
