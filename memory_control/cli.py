from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from model_zoo import available_model_names

from memory_control.core.config import build_config
from memory_control.core.trainer import run_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny PyTorch end-to-end memory-control smoke test.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("out/smoke.csv"))
    p.add_argument("--controller", choices=("static", "headroom", "safe_greedy", "rl"), default=None)
    p.add_argument("--model", choices=available_model_names(), default=None, help="Model architecture to instantiate")
    p.add_argument("--dataset", choices=("synthetic", "cifar10"), default=None, help="Input data source")
    p.add_argument("--data-dir", type=str, default=None, help="Dataset cache directory")
    p.add_argument("--download-dataset", action="store_true", help="Download the dataset when needed")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count for real datasets")
    p.add_argument("--precision", choices=("fp32", "bf16"), default=None, help="CUDA autocast precision")
    p.add_argument("--compile-model", action="store_true", help="Use torch.compile when available")
    p.add_argument("--distributed", action="store_true", help="Enable single-node torch.distributed/DDP mode")
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
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    run_training(cfg, args.output)
