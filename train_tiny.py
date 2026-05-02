#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from model_zoo import available_model_names, build_model, make_synthetic_batch


@dataclass(frozen=True)
class RunConfig:
    steps: int = 80
    seed: int = 0
    device: str = "cpu"
    input_dim: int = 256
    hidden_dim: int = 512
    num_classes: int = 10
    model_name: str = "mlp"
    model_args: dict[str, Any] = field(default_factory=dict)
    initial_micro_batch: int = 16
    initial_grad_accum: int = 1
    max_micro_batch: int = 64
    max_grad_accum: int = 32
    control_interval: int = 10
    budget_mb: float = 96.0
    headroom_margin: float = 0.05
    learning_rate: float = 1e-3
    controller: str = "headroom"


@dataclass(frozen=True)
class Knobs:
    micro_batch: int
    grad_accum_steps: int


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny PyTorch end-to-end memory-control smoke test.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("end_to_end/out/smoke.csv"))
    p.add_argument("--controller", choices=("static", "headroom"), default=None)
    p.add_argument("--model", choices=available_model_names(), default=None, help="Model architecture to instantiate")
    p.add_argument(
        "--model-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one model_args entry, for example --model-arg hidden_dim=1024",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--budget-mb", type=float, default=None)
    return p.parse_args()


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_model_arg_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --model-arg {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid --model-arg {item!r}; empty key")
        overrides[key] = _parse_scalar(value.strip())
    return overrides


def _build_config(args: argparse.Namespace) -> RunConfig:
    raw = _load_config(args.config)
    cfg = RunConfig(**raw)
    if args.controller is not None:
        cfg = replace(cfg, controller=args.controller)
    if args.model is not None:
        cfg = replace(cfg, model_name=args.model, model_args={})
    model_arg_overrides = _parse_model_arg_overrides(args.model_arg)
    if model_arg_overrides:
        cfg = replace(cfg, model_args={**cfg.model_args, **model_arg_overrides})
    if args.steps is not None:
        cfg = replace(cfg, steps=int(args.steps))
    if args.device is not None:
        cfg = replace(cfg, device=str(args.device))
    if args.budget_mb is not None:
        cfg = replace(cfg, budget_mb=float(args.budget_mb))
    return cfg


def _model_args_for_config(cfg: RunConfig) -> dict[str, Any]:
    if cfg.model_args:
        return dict(cfg.model_args)
    if cfg.model_name in {"linear", "mlp"}:
        args: dict[str, Any] = {
            "input_dim": cfg.input_dim,
            "num_classes": cfg.num_classes,
        }
        if cfg.model_name == "mlp":
            args["hidden_dim"] = cfg.hidden_dim
        return args
    return {}


def _estimate_peak_mb(*, model_bytes: int, micro_batch: int, activation_units_per_sample: int) -> float:
    # This CPU smoke test cannot observe CUDA VRAM. The proxy keeps the same
    # control shape as the simulator: larger micro-batches increase peak memory.
    activation_bytes = micro_batch * activation_units_per_sample * 4
    optimizer_bytes = model_bytes * 2
    workspace_bytes = max(8 * 1024 * 1024, activation_bytes * 2)
    return (model_bytes + optimizer_bytes + activation_bytes + workspace_bytes) / (1024 * 1024)


def _maybe_update_knobs(
    *,
    cfg: RunConfig,
    knobs: Knobs,
    peak_mb: float,
    oom: bool,
    comm_frac: float,
) -> tuple[Knobs, bool]:
    if cfg.controller == "static":
        return knobs, False

    limit_mb = cfg.budget_mb * (1.0 - cfg.headroom_margin)
    new_knobs = knobs

    if oom or peak_mb > limit_mb:
        new_knobs = replace(new_knobs, micro_batch=max(1, knobs.micro_batch // 2))
    elif peak_mb < cfg.budget_mb * 0.70 and comm_frac > 0.10:
        new_knobs = replace(
            new_knobs,
            grad_accum_steps=min(cfg.max_grad_accum, knobs.grad_accum_steps + 1),
        )
    elif peak_mb < cfg.budget_mb * 0.45 and knobs.micro_batch < cfg.max_micro_batch:
        new_knobs = replace(
            new_knobs,
            micro_batch=min(cfg.max_micro_batch, knobs.micro_batch * 2),
        )

    return new_knobs, new_knobs != knobs


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch is required for this smoke test. Install it with: python -m pip install torch") from exc

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    model, model_profile = build_model(cfg.model_name, _model_args_for_config(cfg), nn)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_parameters = sum(p.numel() for p in model.parameters())

    knobs = Knobs(
        micro_batch=int(cfg.initial_micro_batch),
        grad_accum_steps=int(cfg.initial_grad_accum),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "samples",
        "step_s",
        "compute_s",
        "comm_s",
        "io_s",
        "peak_mb",
        "budget_mb",
        "oom",
        "micro_batch",
        "grad_accum_steps",
        "next_micro_batch",
        "next_grad_accum_steps",
        "controller",
        "knob_changed",
        "model_name",
        "model_parameters",
        "model_input_shape",
        "loss",
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for step in range(cfg.steps):
            step_knobs = knobs
            started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0

            compute_started = time.perf_counter()
            for _ in range(step_knobs.grad_accum_steps):
                x, y = make_synthetic_batch(model_profile, step_knobs.micro_batch, device, torch)
                logits = model(x)
                loss = F.cross_entropy(logits, y) / max(1, step_knobs.grad_accum_steps)
                loss.backward()
                total_loss += float(loss.detach().cpu()) * max(1, step_knobs.grad_accum_steps)

            opt.step()
            compute_s = time.perf_counter() - compute_started

            # Local single-process proxy for synchronization. Higher accumulation
            # amortizes fixed communication overhead, matching the SimGrid insight.
            comm_s = 0.0008 + 0.004 / math.sqrt(max(1, step_knobs.grad_accum_steps))
            io_s = 0.0
            step_s = (time.perf_counter() - started) + comm_s + io_s

            peak_mb = _estimate_peak_mb(
                model_bytes=model_bytes,
                micro_batch=step_knobs.micro_batch,
                activation_units_per_sample=model_profile.activation_units_per_sample,
            )
            oom = peak_mb > cfg.budget_mb
            comm_frac = comm_s / max(step_s, 1e-12)
            knob_changed = False

            if (step + 1) % cfg.control_interval == 0:
                knobs, knob_changed = _maybe_update_knobs(
                    cfg=cfg,
                    knobs=step_knobs,
                    peak_mb=peak_mb,
                    oom=oom,
                    comm_frac=comm_frac,
                )

            writer.writerow(
                {
                    "step": step + 1,
                    "samples": step_knobs.micro_batch * step_knobs.grad_accum_steps,
                    "step_s": f"{step_s:.8f}",
                    "compute_s": f"{compute_s:.8f}",
                    "comm_s": f"{comm_s:.8f}",
                    "io_s": f"{io_s:.8f}",
                    "peak_mb": f"{peak_mb:.4f}",
                    "budget_mb": f"{cfg.budget_mb:.4f}",
                    "oom": int(oom),
                    "micro_batch": step_knobs.micro_batch,
                    "grad_accum_steps": step_knobs.grad_accum_steps,
                    "next_micro_batch": knobs.micro_batch,
                    "next_grad_accum_steps": knobs.grad_accum_steps,
                    "controller": cfg.controller,
                    "knob_changed": int(knob_changed),
                    "model_name": model_profile.name,
                    "model_parameters": model_parameters,
                    "model_input_shape": "x".join(str(dim) for dim in model_profile.input_shape),
                    "loss": f"{total_loss:.6f}",
                }
            )

    print(f"wrote {args.output}")
    print(
        "model="
        f"{model_profile.name} params={model_parameters} "
        f"input_shape={model_profile.input_shape} description={model_profile.description}"
    )


if __name__ == "__main__":
    main()
