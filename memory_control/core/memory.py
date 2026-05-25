from __future__ import annotations

import math
from dataclasses import replace

from memory_control.core.config import RunConfig
from memory_control.core.types import Knobs


def estimate_peak_mb(*, model_bytes: int, micro_batch: int, activation_units_per_sample: int) -> float:
    # This CPU smoke test cannot observe CUDA VRAM. The proxy keeps the same
    # control shape as the simulator: larger micro-batches increase peak memory.
    activation_bytes = micro_batch * activation_units_per_sample * 4
    optimizer_bytes = model_bytes * 2
    workspace_bytes = max(8 * 1024 * 1024, activation_bytes * 2)
    return (model_bytes + optimizer_bytes + activation_bytes + workspace_bytes) / (1024 * 1024)


def clip_knobs(cfg: RunConfig, knobs: Knobs) -> Knobs:
    return Knobs(
        micro_batch=max(1, min(int(cfg.max_micro_batch), int(knobs.micro_batch))),
        grad_accum_steps=max(1, min(int(cfg.max_grad_accum), int(knobs.grad_accum_steps))),
    )


def safer_knobs(cfg: RunConfig, knobs: Knobs) -> Knobs:
    if knobs.micro_batch > 1:
        new_micro = max(1, knobs.micro_batch // 2)
        effective_batch = max(1, knobs.micro_batch * knobs.grad_accum_steps)
        new_accum = min(int(cfg.max_grad_accum), max(knobs.grad_accum_steps, math.ceil(effective_batch / new_micro)))
        return clip_knobs(cfg, Knobs(new_micro, new_accum))
    return clip_knobs(cfg, Knobs(knobs.micro_batch, min(int(cfg.max_grad_accum), knobs.grad_accum_steps + 1)))


def predict_peak_mb(cfg: RunConfig, *, model_bytes: int, activation_units_per_sample: int, knobs: Knobs) -> float:
    return estimate_peak_mb(
        model_bytes=model_bytes,
        micro_batch=clip_knobs(cfg, knobs).micro_batch,
        activation_units_per_sample=activation_units_per_sample,
    )


def increase_grad_accum(cfg: RunConfig, knobs: Knobs) -> Knobs:
    return replace(knobs, grad_accum_steps=min(cfg.max_grad_accum, knobs.grad_accum_steps + 1))


def increase_micro_batch(cfg: RunConfig, knobs: Knobs) -> Knobs:
    return replace(knobs, micro_batch=min(cfg.max_micro_batch, knobs.micro_batch * 2))
