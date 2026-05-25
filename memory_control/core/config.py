from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


VALID_CONTROLLERS = {"static", "headroom", "safe_greedy", "rl"}
VALID_DATASETS = {"synthetic", "cifar10"}
VALID_PRECISIONS = {"fp32", "bf16"}


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
    dataset_name: str = "synthetic"
    data_dir: str = "data"
    download_dataset: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    precision: str = "fp32"
    compile_model: bool = False
    distributed: bool = False
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


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r") as f:
        return json.load(f)


def parse_scalar(value: str) -> Any:
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


def parse_model_arg_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --model-arg {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid --model-arg {item!r}; empty key")
        overrides[key] = parse_scalar(value.strip())
    return overrides


def build_config(args: argparse.Namespace) -> RunConfig:
    raw = load_config(args.config)
    cfg = RunConfig(**raw)
    if args.controller is not None:
        cfg = replace(cfg, controller=args.controller)
    if args.model is not None:
        cfg = replace(cfg, model_name=args.model, model_args={})
    if args.dataset is not None:
        cfg = replace(cfg, dataset_name=str(args.dataset))
    if args.data_dir is not None:
        cfg = replace(cfg, data_dir=str(args.data_dir))
    if args.download_dataset:
        cfg = replace(cfg, download_dataset=True)
    if args.num_workers is not None:
        cfg = replace(cfg, num_workers=int(args.num_workers))
    if args.precision is not None:
        cfg = replace(cfg, precision=str(args.precision))
    if args.compile_model:
        cfg = replace(cfg, compile_model=True)
    if args.distributed:
        cfg = replace(cfg, distributed=True)
    model_arg_overrides = parse_model_arg_overrides(args.model_arg)
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
    if cfg.controller not in VALID_CONTROLLERS:
        raise SystemExit(f"unknown controller={cfg.controller!r}")
    if cfg.controller == "rl" and not cfg.rl_checkpoint:
        raise SystemExit("--controller rl requires --rl-checkpoint or rl_checkpoint in the config")
    if cfg.dataset_name not in VALID_DATASETS:
        raise SystemExit(f"unknown dataset_name={cfg.dataset_name!r}")
    if cfg.precision not in VALID_PRECISIONS:
        raise SystemExit(f"unknown precision={cfg.precision!r}")
    return cfg


def model_args_for_config(cfg: RunConfig) -> dict[str, Any]:
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
