from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from memory_control.core.config import RunConfig
from memory_control.core.memory import (
    clip_knobs,
    increase_grad_accum,
    increase_micro_batch,
    predict_peak_mb,
    safer_knobs,
)
from memory_control.core.types import ControllerState, Knobs


def _rl_package_src() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root.parent / "rl_memory_agent" / "src"


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


def _sync_rl_config_from_knobs(state: ControllerState, knobs: Knobs):
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


def maybe_update_knobs(
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
            new_knobs = safer_knobs(cfg, knobs)
        elif peak_mb < cfg.budget_mb * 0.70 and comm_frac > 0.10:
            new_knobs = increase_grad_accum(cfg, new_knobs)
        elif peak_mb < cfg.budget_mb * 0.45 and knobs.micro_batch < cfg.max_micro_batch:
            new_knobs = increase_micro_batch(cfg, new_knobs)
        new_knobs = clip_knobs(cfg, new_knobs)
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
        new_knobs = clip_knobs(cfg, Knobs(int(proposed.micro_batch), int(proposed.grad_accum_steps)))
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
        new_knobs = safer_knobs(cfg, rollback)
        state.in_trial = True
        return new_knobs, new_knobs != knobs

    candidate = knobs
    if comm_frac > 0.10 and knobs.grad_accum_steps < cfg.max_grad_accum:
        candidate = replace(candidate, grad_accum_steps=knobs.grad_accum_steps + 1)
    elif peak_mb < cfg.budget_mb * 0.70 and knobs.micro_batch < cfg.max_micro_batch:
        candidate = replace(candidate, micro_batch=knobs.micro_batch * 2)

    candidate = clip_knobs(cfg, candidate)
    if candidate == knobs:
        return knobs, False

    predicted_peak = predict_peak_mb(
        cfg,
        model_bytes=model_bytes,
        activation_units_per_sample=activation_units_per_sample,
        knobs=candidate,
    )
    if predicted_peak > limit_mb:
        if comm_frac > 0.10 and knobs.grad_accum_steps < cfg.max_grad_accum:
            conservative = clip_knobs(cfg, replace(knobs, grad_accum_steps=knobs.grad_accum_steps + 1))
            if conservative != knobs:
                state.in_trial = True
                return conservative, True
        return knobs, False

    state.in_trial = True
    return candidate, True
