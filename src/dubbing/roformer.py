"""BS-Roformer vocal separation via `audio-separator` (runs in .venv-roformer).

Standalone so it stays in its own environment. Reads an input wav, writes
standardized ``vocals.wav`` and ``instrumental.wav`` into --outdir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    args = ap.parse_args(argv)

    from audio_separator.separator import Separator

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(outdir))
    sep.load_model(model_filename=args.model)
    outputs = sep.separate(args.input)  # basenames written into outdir

    def _find(*keys):
        for name in outputs:
            low = name.lower()
            if any(k in low for k in keys):
                return outdir / name
        return None

    voc = _find("vocal")
    inst = _find("instrument", "no_vocal", "accompaniment")
    if not voc or not inst:
        print(f"[roformer] could not identify stems in {outputs}", file=sys.stderr)
        return 1

    v, sr = sf.read(str(voc))
    i, _ = sf.read(str(inst))
    sf.write(str(outdir / "vocals.wav"), np.asarray(v), sr)
    sf.write(str(outdir / "instrumental.wav"), np.asarray(i), sr)
    print(f"[roformer] wrote vocals.wav + instrumental.wav (sr={sr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
