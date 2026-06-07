"""GPU VRAM helpers for the phased, memory-budgeted pipeline."""

from __future__ import annotations

import shutil
import subprocess


def _smi(query: str, index: int = 0):
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, check=True,
        )
        return int(float(out.stdout.strip().splitlines()[0]))
    except Exception:
        return None


def free_mb(index: int = 0):
    return _smi("memory.free", index)


def total_mb(index: int = 0):
    return _smi("memory.total", index)


def used_mb(index: int = 0):
    return _smi("memory.used", index)


def suggest_llm_context(weights_gb: float = 19.0, index: int = 0,
                        reserve_mb: int = 2000, min_ctx: int = 2048,
                        max_ctx: int = 8192) -> int:
    """Heuristic target context length for the LLM given FREE VRAM.

    Uses currently-free VRAM (so other apps' usage is accounted for) minus the
    model weights and a safety reserve. KV cache for a ~31B model is on the order
    of ~0.3 MB/token with LM Studio's quantized cache. The cap is deliberately
    modest: translation windows are short, and an oversized context just forces
    CPU offload (which is catastrophically slow). LM Studio still fits at load.
    """
    free = free_mb(index)
    if not free:
        return min_ctx
    budget_kv = free - int(weights_gb * 1024) - reserve_mb
    if budget_kv <= 0:
        return min_ctx
    ctx = int(budget_kv / 0.30)
    return max(min_ctx, min(max_ctx, (ctx // 1024) * 1024))
