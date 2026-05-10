#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
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
    initial_bucket_cap_mb: int = 25
    comm_base_s: float = 0.0008
    comm_scale_s: float = 0.004
    io_s: float = 0.0
    safe_greedy_min_improvement: float = 0.005
    rl_checkpoint: str | None = None
    rl_action_profile: str = "fast_only"
    rl_window_len: int = 20
    rl_device: str = "cpu"


@dataclass(frozen=True)
class Knobs:
    micro_batch: int
    grad_accum_steps: int


@dataclass
class ControllerState:
    best_knobs: Knobs | None = None
    best_throughput: float = 0.0
    in_trial: bool = False
    rl_algo: Any | None = None
    rl_action_space: Any | None = None
    rl_state_builder: Any | None = None
    rl_window: Any | None = None
    rl_safety: Any | None = None
    rl_knob_config: Any | None = None
    rl_modules: dict[str, Any] = field(default_factory=dict)
    last_action_name: str = ""
    last_safety_allowed: bool = True
    last_safety_reason: str = ""


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny PyTorch end-to-end memory-control smoke test.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("out/smoke.csv"))
    p.add_argument("--controller", choices=("static", "headroom", "safe_greedy", "rl"), default=None)
    p.add_argument("--model", choices=available_model_names(), default=None, help="Model architecture to instantiate")
    p.add_argument(
        "--model-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one model_args entry, for example --model-arg hidden_dim=1024",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--budget-mb", type=float, default=None)
    p.add_argument("--rl-checkpoint", type=str, default=None, help="PPO checkpoint to use when --controller rl")
    p.add_argument("--rl-action-profile", choices=("all", "fast_only"), default=None)
    p.add_argument("--rl-device", type=str, default=None, help="Device used for RL policy inference")
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
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if args.device is not None:
        cfg = replace(cfg, device=str(args.device))
    if args.budget_mb is not None:
        cfg = replace(cfg, budget_mb=float(args.budget_mb))
    if args.rl_checkpoint is not None:
        cfg = replace(cfg, rl_checkpoint=str(args.rl_checkpoint))
    if args.rl_action_profile is not None:
        cfg = replace(cfg, rl_action_profile=str(args.rl_action_profile))
    if args.rl_device is not None:
        cfg = replace(cfg, rl_device=str(args.rl_device))
    if cfg.controller not in {"static", "headroom", "safe_greedy", "rl"}:
        raise SystemExit(f"unknown controller={cfg.controller!r}")
    if cfg.controller == "rl" and not cfg.rl_checkpoint:
        raise SystemExit("--controller rl requires --rl-checkpoint or rl_checkpoint in the config")
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


def _clip_knobs(cfg: RunConfig, knobs: Knobs) -> Knobs:
    return Knobs(
        micro_batch=max(1, min(int(cfg.max_micro_batch), int(knobs.micro_batch))),
        grad_accum_steps=max(1, min(int(cfg.max_grad_accum), int(knobs.grad_accum_steps))),
    )


def _safer_knobs(cfg: RunConfig, knobs: Knobs) -> Knobs:
    if knobs.micro_batch > 1:
        new_micro = max(1, knobs.micro_batch // 2)
        effective_batch = max(1, knobs.micro_batch * knobs.grad_accum_steps)
        new_accum = min(int(cfg.max_grad_accum), max(knobs.grad_accum_steps, math.ceil(effective_batch / new_micro)))
        return _clip_knobs(cfg, Knobs(new_micro, new_accum))
    return _clip_knobs(cfg, Knobs(knobs.micro_batch, min(int(cfg.max_grad_accum), knobs.grad_accum_steps + 1)))


def _predict_peak_mb(cfg: RunConfig, *, model_bytes: int, activation_units_per_sample: int, knobs: Knobs) -> float:
    return _estimate_peak_mb(
        model_bytes=model_bytes,
        micro_batch=_clip_knobs(cfg, knobs).micro_batch,
        activation_units_per_sample=activation_units_per_sample,
    )


def _rl_package_src() -> Path:
    return Path(__file__).resolve().parents[1] / "rl_memory_agent" / "src"


def _ensure_rl_controller(cfg: RunConfig, state: ControllerState, knobs: Knobs) -> None:
    if state.rl_algo is not None:
        return

    checkpoint = Path(str(cfg.rl_checkpoint)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    if not checkpoint.exists():
        raise SystemExit(f"RL checkpoint not found: {checkpoint}")

    src = _rl_package_src()
    if not src.exists():
        raise SystemExit(f"rl_memory_agent source tree not found: {src}")
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)

    try:
        from rl_memory_agent.knobs import KnobActionSpace, KnobConfig, KnobConstraints
        from rl_memory_agent.ppo_lagrangian import LagrangianConfig, LagrangianPPO, PPOConfig
        from rl_memory_agent.safety import SafetyShield
        from rl_memory_agent.state import StateBuilder
        from rl_memory_agent.telemetry import TelemetrySample, TelemetryWindow
    except ImportError as exc:
        raise SystemExit(f"could not import rl_memory_agent from {src}: {exc}") from exc

    constraints = KnobConstraints(
        micro_batch_min=1,
        micro_batch_max=int(cfg.max_micro_batch),
        grad_accum_min=1,
        grad_accum_max=int(cfg.max_grad_accum),
    )
    action_space = KnobActionSpace(constraints, action_profile=str(cfg.rl_action_profile))
    state_builder = StateBuilder(budget_mb=float(cfg.budget_mb), action_space=action_space)
    algo = LagrangianPPO(
        obs_dim=len(state_builder.spec.names),
        n_actions=action_space.n,
        ppo=PPOConfig(),
        lagrangian=LagrangianConfig(cost_limits=(0.0,)),
        device=str(cfg.rl_device),
    )
    try:
        extra = algo.load_checkpoint(str(checkpoint))
    except RuntimeError as exc:
        raise SystemExit(
            "failed to load RL checkpoint. Check that rl_action_profile matches the checkpoint "
            f"(current={cfg.rl_action_profile!r}) and that the checkpoint uses the same state/action dimensions: {exc}"
        ) from exc

    checkpoint_profile = str(extra.get("action_profile", "")) if isinstance(extra, dict) else ""
    if checkpoint_profile and checkpoint_profile != str(cfg.rl_action_profile):
        raise SystemExit(
            f"RL checkpoint action_profile={checkpoint_profile!r} does not match "
            f"rl_action_profile={cfg.rl_action_profile!r}"
        )

    state.rl_algo = algo
    state.rl_action_space = action_space
    state.rl_state_builder = state_builder
    state.rl_window = TelemetryWindow(maxlen=int(cfg.rl_window_len))
    state.rl_safety = SafetyShield(budget_mb=float(cfg.budget_mb), headroom_margin=float(cfg.headroom_margin))
    state.rl_knob_config = KnobConfig(
        micro_batch=int(knobs.micro_batch),
        grad_accum_steps=int(knobs.grad_accum_steps),
        bucket_cap_mb=int(cfg.initial_bucket_cap_mb),
    )
    state.rl_modules = {
        "KnobConfig": KnobConfig,
        "TelemetrySample": TelemetrySample,
    }
    state.last_action_name = "checkpoint_loaded"
    state.last_safety_allowed = True
    state.last_safety_reason = f"loaded:{checkpoint}"


def _sync_rl_config_from_knobs(state: ControllerState, knobs: Knobs) -> Any:
    knob_config = state.rl_knob_config
    if knob_config is None:
        raise RuntimeError("RL knob config was not initialized")
    KnobConfig = state.rl_modules["KnobConfig"]
    return KnobConfig(
        micro_batch=int(knobs.micro_batch),
        grad_accum_steps=int(knobs.grad_accum_steps),
        activation_checkpointing=bool(knob_config.activation_checkpointing),
        precision=str(knob_config.precision),
        sharding=str(knob_config.sharding),
        bucket_cap_mb=int(knob_config.bucket_cap_mb),
        ckpt_interval_steps=int(knob_config.ckpt_interval_steps),
    )


def _maybe_update_knobs(
    *,
    cfg: RunConfig,
    knobs: Knobs,
    state: ControllerState,
    step: int,
    peak_mb: float,
    allocated_mb: float,
    reserved_mb: float,
    oom: bool,
    compute_s: float,
    comm_s: float,
    io_s: float,
    step_s: float,
    comm_frac: float,
    throughput: float,
    model_bytes: int,
    activation_units_per_sample: int,
) -> tuple[Knobs, bool]:
    if cfg.controller == "static":
        return knobs, False

    limit_mb = cfg.budget_mb * (1.0 - cfg.headroom_margin)
    safe = (not oom) and peak_mb <= limit_mb

    if cfg.controller == "headroom":
        new_knobs = knobs
        if not safe:
            new_knobs = _safer_knobs(cfg, knobs)
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
        new_knobs = _clip_knobs(cfg, new_knobs)
        return new_knobs, new_knobs != knobs

    if cfg.controller == "rl":
        _ensure_rl_controller(cfg, state, knobs)
        TelemetrySample = state.rl_modules["TelemetrySample"]
        sample = TelemetrySample(
            step=int(step),
            vram_allocated_mb=float(allocated_mb),
            vram_reserved_mb=float(reserved_mb),
            vram_peak_mb=float(peak_mb),
            step_time_s=float(step_s),
            compute_time_s=float(compute_s),
            comm_time_s=float(comm_s),
            io_time_s=float(io_s),
            oom=bool(oom),
            restart=False,
        )
        assert state.rl_window is not None
        assert state.rl_state_builder is not None
        assert state.rl_algo is not None
        assert state.rl_action_space is not None
        assert state.rl_safety is not None

        current_config = _sync_rl_config_from_knobs(state, knobs)
        state.rl_window.append(sample)
        obs = state.rl_state_builder.build(state.rl_window, current_config)

        if safe:
            state.rl_safety.record_safe(current_config)

        action_id, _log_prob, _value = state.rl_algo.select_action(obs)
        action_names = state.rl_action_space.names()
        state.last_action_name = action_names[int(action_id)] if 0 <= int(action_id) < len(action_names) else str(action_id)
        proposed = state.rl_action_space.apply(int(action_id), current_config)

        safety_res = state.rl_safety.check(last=sample, current=current_config, proposed=proposed)
        state.last_safety_allowed = bool(safety_res.allowed)
        state.last_safety_reason = str(safety_res.reason)
        if not safety_res.allowed:
            proposed = current_config

        if oom:
            safe_config = state.rl_safety.last_safe_config()
            if safe_config is not None:
                proposed = safe_config
                state.last_action_name = "rollback_last_safe"
                state.last_safety_allowed = True
                state.last_safety_reason = "rollback_after_oom"

        state.rl_knob_config = proposed
        new_knobs = _clip_knobs(cfg, Knobs(int(proposed.micro_batch), int(proposed.grad_accum_steps)))
        return new_knobs, new_knobs != knobs

    if cfg.controller != "safe_greedy":
        raise ValueError(f"unknown controller={cfg.controller!r}")

    if safe and (state.best_knobs is None or throughput > state.best_throughput * 1.01):
        state.best_knobs = knobs
        state.best_throughput = throughput
        state.in_trial = False
    elif (
        safe
        and state.in_trial
        and state.best_knobs is not None
        and throughput <= state.best_throughput * (1.0 + float(cfg.safe_greedy_min_improvement))
    ):
        state.in_trial = False
        return state.best_knobs, state.best_knobs != knobs

    if not safe:
        rollback = state.best_knobs if state.best_knobs is not None else knobs
        new_knobs = _safer_knobs(cfg, rollback)
        state.in_trial = True
        return new_knobs, new_knobs != knobs

    candidate = knobs
    if comm_frac > 0.10 and knobs.grad_accum_steps < cfg.max_grad_accum:
        candidate = replace(candidate, grad_accum_steps=knobs.grad_accum_steps + 1)
    elif peak_mb < cfg.budget_mb * 0.70 and knobs.micro_batch < cfg.max_micro_batch:
        candidate = replace(candidate, micro_batch=knobs.micro_batch * 2)

    candidate = _clip_knobs(cfg, candidate)
    if candidate == knobs:
        return knobs, False

    predicted_peak = _predict_peak_mb(
        cfg,
        model_bytes=model_bytes,
        activation_units_per_sample=activation_units_per_sample,
        knobs=candidate,
    )
    if predicted_peak > limit_mb:
        if comm_frac > 0.10 and knobs.grad_accum_steps < cfg.max_grad_accum:
            conservative = _clip_knobs(cfg, replace(knobs, grad_accum_steps=knobs.grad_accum_steps + 1))
            if conservative != knobs:
                state.in_trial = True
                return conservative, True
        return knobs, False

    state.in_trial = True
    return candidate, True


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


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
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    model, model_profile = build_model(cfg.model_name, _model_args_for_config(cfg), nn)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_parameters = sum(p.numel() for p in model.parameters())

    knobs = Knobs(
        micro_batch=int(cfg.initial_micro_batch),
        grad_accum_steps=int(cfg.initial_grad_accum),
    )
    controller_state = ControllerState()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "samples",
        "step_s",
        "compute_s",
        "comm_s",
        "io_s",
        "comm_frac",
        "io_frac",
        "peak_mb",
        "proxy_peak_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "memory_source",
        "budget_mb",
        "safe_limit_mb",
        "oom",
        "micro_batch",
        "grad_accum_steps",
        "next_micro_batch",
        "next_grad_accum_steps",
        "controller",
        "knob_changed",
        "controller_action",
        "safety_allowed",
        "safety_reason",
        "model_name",
        "model_parameters",
        "model_input_shape",
        "loss",
        "effective_samples_per_s",
        "best_safe_throughput",
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for step in range(cfg.steps):
            step_knobs = knobs
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            torch_oom = False

            compute_started = time.perf_counter()
            try:
                for _ in range(step_knobs.grad_accum_steps):
                    x, y = make_synthetic_batch(model_profile, step_knobs.micro_batch, device, torch)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y) / max(1, step_knobs.grad_accum_steps)
                    loss.backward()
                    total_loss += float(loss.detach().cpu()) * max(1, step_knobs.grad_accum_steps)

                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except RuntimeError as exc:
                if not (device.type == "cuda" and _is_cuda_oom(exc)):
                    raise
                torch_oom = True
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                total_loss = float("nan")
            compute_s = time.perf_counter() - compute_started

            # Local single-process proxy for synchronization. Higher accumulation
            # amortizes fixed communication overhead, matching the SimGrid insight.
            comm_s = float(cfg.comm_base_s) + float(cfg.comm_scale_s) / math.sqrt(max(1, step_knobs.grad_accum_steps))
            io_s = float(cfg.io_s)
            step_s = (time.perf_counter() - started) + comm_s + io_s

            proxy_peak_mb = _estimate_peak_mb(
                model_bytes=model_bytes,
                micro_batch=step_knobs.micro_batch,
                activation_units_per_sample=model_profile.activation_units_per_sample,
            )
            cuda_peak_allocated_mb = 0.0
            cuda_peak_reserved_mb = 0.0
            memory_source = "proxy"
            peak_mb = proxy_peak_mb
            if device.type == "cuda":
                cuda_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                cuda_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
                peak_mb = max(cuda_peak_reserved_mb, cuda_peak_allocated_mb)
                memory_source = "cuda"

            oom = torch_oom or peak_mb > cfg.budget_mb
            allocated_mb = cuda_peak_allocated_mb if memory_source == "cuda" else peak_mb * 0.85
            reserved_mb = cuda_peak_reserved_mb if memory_source == "cuda" else peak_mb * 0.95
            comm_frac = comm_s / max(step_s, 1e-12)
            io_frac = io_s / max(step_s, 1e-12)
            effective_samples = 0 if oom else step_knobs.micro_batch * step_knobs.grad_accum_steps
            effective_samples_per_s = effective_samples / max(step_s, 1e-12)
            knob_changed = False

            if oom or (step + 1) % cfg.control_interval == 0:
                knobs, knob_changed = _maybe_update_knobs(
                    cfg=cfg,
                    knobs=step_knobs,
                    state=controller_state,
                    step=step + 1,
                    peak_mb=peak_mb,
                    allocated_mb=allocated_mb,
                    reserved_mb=reserved_mb,
                    oom=oom,
                    compute_s=compute_s,
                    comm_s=comm_s,
                    io_s=io_s,
                    step_s=step_s,
                    comm_frac=comm_frac,
                    throughput=effective_samples_per_s,
                    model_bytes=model_bytes,
                    activation_units_per_sample=model_profile.activation_units_per_sample,
                )

            writer.writerow(
                {
                    "step": step + 1,
                    "samples": effective_samples,
                    "step_s": f"{step_s:.8f}",
                    "compute_s": f"{compute_s:.8f}",
                    "comm_s": f"{comm_s:.8f}",
                    "io_s": f"{io_s:.8f}",
                    "comm_frac": f"{comm_frac:.8f}",
                    "io_frac": f"{io_frac:.8f}",
                    "peak_mb": f"{peak_mb:.4f}",
                    "proxy_peak_mb": f"{proxy_peak_mb:.4f}",
                    "cuda_peak_allocated_mb": f"{cuda_peak_allocated_mb:.4f}",
                    "cuda_peak_reserved_mb": f"{cuda_peak_reserved_mb:.4f}",
                    "memory_source": memory_source,
                    "budget_mb": f"{cfg.budget_mb:.4f}",
                    "safe_limit_mb": f"{cfg.budget_mb * (1.0 - cfg.headroom_margin):.4f}",
                    "oom": int(oom),
                    "micro_batch": step_knobs.micro_batch,
                    "grad_accum_steps": step_knobs.grad_accum_steps,
                    "next_micro_batch": knobs.micro_batch,
                    "next_grad_accum_steps": knobs.grad_accum_steps,
                    "controller": cfg.controller,
                    "knob_changed": int(knob_changed),
                    "controller_action": controller_state.last_action_name,
                    "safety_allowed": int(controller_state.last_safety_allowed),
                    "safety_reason": controller_state.last_safety_reason,
                    "model_name": model_profile.name,
                    "model_parameters": model_parameters,
                    "model_input_shape": "x".join(str(dim) for dim in model_profile.input_shape),
                    "loss": f"{total_loss:.6f}" if math.isfinite(total_loss) else "nan",
                    "effective_samples_per_s": f"{effective_samples_per_s:.8f}",
                    "best_safe_throughput": f"{controller_state.best_throughput:.8f}",
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
