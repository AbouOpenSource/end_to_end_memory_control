from __future__ import annotations

import contextlib
from typing import Any

from memory_control.core.config import RunConfig


def autocast_context(cfg: RunConfig, device: Any, torch: Any) -> Any:
    if device.type != "cuda" or cfg.precision == "fp32":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg
