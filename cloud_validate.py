#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ValidationCase:
    name: str
    config: str
    controllers: str
    baseline: str
    needs_dataset: bool = False
    needs_rl: bool = False


BASE_CASES = (
    ValidationCase(
        name="synthetic_cuda",
        config="configs/cloud_synthetic_cuda.json",
        controllers="static,headroom,safe_greedy",
        baseline="static",
    ),
    ValidationCase(
        name="cifar10_cuda",
        config="configs/cloud_cifar10_cuda.json",
        controllers="static,headroom,safe_greedy",
        baseline="static",
        needs_dataset=True,
    ),
    ValidationCase(
        name="cifar10_tight_cuda",
        config="configs/cloud_cifar10_tight_cuda.json",
        controllers="static,headroom,safe_greedy",
        baseline="static",
        needs_dataset=True,
    ),
)

RL_CASE = ValidationCase(
    name="rl_transfer_cifar10_tight",
    config="configs/cloud_cifar10_tight_cuda.json",
    controllers="safe_greedy,rl",
    baseline="safe_greedy",
    needs_dataset=True,
    needs_rl=True,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run cloud validation matrices for the end-to-end memory controller.")
    p.add_argument("--preset", choices=("all", "synthetic", "cifar10", "tight", "rl"), default="all")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-cpu", action="store_true", help="Allow execution without CUDA for debugging only")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=None, help="Override steps in every config")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--distributed", action="store_true", help="Run each case with torchrun/DDP")
    p.add_argument("--nproc-per-node", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--download-dataset", action="store_true")
    p.add_argument("--rl-checkpoint", type=Path, default=None)
    p.add_argument("--rl-action-profile", choices=("all", "fast_only"), default="fast_only")
    p.add_argument("--rl-device", default="cpu")
    p.add_argument("--skip-report", action="store_true")
    return p.parse_args()


def _select_cases(preset: str, rl_checkpoint: Path | None) -> list[ValidationCase]:
    if preset == "synthetic":
        return [BASE_CASES[0]]
    if preset == "cifar10":
        return [BASE_CASES[1]]
    if preset == "tight":
        return [BASE_CASES[2]]
    if preset == "rl":
        return [RL_CASE]
    cases = list(BASE_CASES)
    if rl_checkpoint is not None:
        cases.append(RL_CASE)
    return cases


def _require_cuda(device: str, allow_cpu: bool) -> None:
    if device != "cuda" or allow_cpu:
        return
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required. Install with: python -m pip install -r requirements-cloud.txt") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Re-run with --allow-cpu only for local debugging.")


def _hardware_snapshot() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch": "not-installed"}
    info: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_mb"] = int(props.total_memory / (1024 * 1024))
    return info


def _run_case(script_dir: Path, case: ValidationCase, args: argparse.Namespace, out_root: Path) -> None:
    if case.needs_rl and args.rl_checkpoint is None:
        raise SystemExit("--preset rl requires --rl-checkpoint")

    case_out = out_root / case.name
    cmd = [
        sys.executable,
        str(script_dir / "run_matrix.py"),
        "--config",
        str(script_dir / case.config),
        "--controllers",
        case.controllers,
        "--seeds",
        str(args.seeds),
        "--baseline",
        case.baseline,
        "--out-dir",
        str(case_out),
        "--device",
        str(args.device),
        "--precision",
        str(args.precision),
        "--data-dir",
        str(args.data_dir),
    ]
    if args.steps is not None:
        cmd.extend(["--steps", str(int(args.steps))])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(int(args.num_workers))])
    if args.download_dataset and case.needs_dataset:
        cmd.append("--download-dataset")
    if args.compile_model:
        cmd.append("--compile-model")
    if args.distributed:
        cmd.append("--distributed")
        if args.nproc_per_node is not None:
            cmd.extend(["--nproc-per-node", str(int(args.nproc_per_node))])
    if case.needs_rl:
        cmd.extend(["--rl-checkpoint", str(args.rl_checkpoint)])
        cmd.extend(["--rl-action-profile", str(args.rl_action_profile)])
        cmd.extend(["--rl-device", str(args.rl_device)])

    print(f"\n=== {case.name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(script_dir), check=True)


def main() -> None:
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    _require_cuda(str(args.device), bool(args.allow_cpu))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir if args.out_dir is not None else Path("out") / "cloud_validation" / timestamp
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _select_cases(str(args.preset), args.rl_checkpoint)
    metadata = {
        "created_at": timestamp,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hardware": _hardware_snapshot(),
        "cases": [case.__dict__ for case in cases],
    }
    with (out_root / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    for case in cases:
        _run_case(script_dir, case, args, out_root)

    if not args.skip_report:
        report_cmd = [sys.executable, str(script_dir / "summarize_validation.py"), str(out_root)]
        subprocess.run(report_cmd, cwd=str(script_dir), check=True)

    print(f"\nvalidation outputs: {out_root}")


if __name__ == "__main__":
    main()
