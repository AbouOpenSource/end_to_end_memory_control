#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _mean_ci95(values: Iterable[float]) -> tuple[float, float, float, int]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = statistics.fmean(clean)
    if n == 1:
        return mean, float("nan"), float("nan"), 1
    sd = statistics.stdev(clean)
    t = T_CRIT_95.get(n - 1, 1.96)
    return mean, sd, t * sd / math.sqrt(n), n


def _summarize_telemetry(path: Path) -> dict[str, object]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty telemetry: {path}")

    step_times = [_as_float(row, "step_s") for row in rows]
    peaks = [_as_float(row, "peak_mb") for row in rows]
    total_step_s = sum(step_times)
    total_samples = sum(_as_float(row, "samples") for row in rows)
    oom_count = sum(int(float(row.get("oom", "0") or 0)) for row in rows)
    knob_changes = sum(int(float(row.get("knob_changed", "0") or 0)) for row in rows)
    final = rows[-1]

    return {
        "rows": len(rows),
        "telemetry_csv": str(path),
        "controller": final.get("controller", ""),
        "model_name": final.get("model_name", ""),
        "model_parameters": final.get("model_parameters", ""),
        "model_input_shape": final.get("model_input_shape", ""),
        "memory_source": final.get("memory_source", ""),
        "total_samples": total_samples,
        "total_step_s": total_step_s,
        "throughput_samples_per_s": total_samples / max(total_step_s, 1e-12),
        "mean_step_s": statistics.fmean(step_times),
        "p95_step_s": _percentile(step_times, 95),
        "mean_peak_mb": statistics.fmean(peaks),
        "max_peak_mb": max(peaks),
        "oom_rate": oom_count / max(len(rows), 1),
        "knob_changes": knob_changes,
        "final_micro_batch": final.get("next_micro_batch", final.get("micro_batch", "")),
        "final_grad_accum_steps": final.get("next_grad_accum_steps", final.get("grad_accum_steps", "")),
        "final_controller_action": final.get("controller_action", ""),
        "final_safety_reason": final.get("safety_reason", ""),
        "final_loss": final.get("loss", ""),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(summary_rows: list[dict[str, object]], baseline: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = ("throughput_samples_per_s", "mean_step_s", "max_peak_mb", "oom_rate", "knob_changes")
    by_controller: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_controller_seed: dict[tuple[str, int], dict[str, object]] = {}
    for row in summary_rows:
        controller = str(row["controller"])
        seed = int(row["seed"])
        by_controller[controller].append(row)
        by_controller_seed[(controller, seed)] = row

    aggregate_rows: list[dict[str, object]] = []
    for controller, rows in sorted(by_controller.items()):
        for metric in metrics:
            mean, sd, ci95, n = _mean_ci95(float(row[metric]) for row in rows)
            aggregate_rows.append(
                {
                    "controller": controller,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci95": ci95,
                }
            )

    speedup_rows: list[dict[str, object]] = []
    baseline_seeds = {
        seed
        for controller, seed in by_controller_seed
        if controller == baseline
    }
    for controller in sorted(by_controller):
        if controller == baseline:
            continue
        paired = []
        for seed in sorted(baseline_seeds):
            base = by_controller_seed.get((baseline, seed))
            cur = by_controller_seed.get((controller, seed))
            if not base or not cur:
                continue
            base_value = float(base["throughput_samples_per_s"])
            cur_value = float(cur["throughput_samples_per_s"])
            if base_value > 0:
                paired.append(100.0 * (cur_value - base_value) / base_value)
        mean, sd, ci95, n = _mean_ci95(paired)
        speedup_rows.append(
            {
                "controller": controller,
                "baseline": baseline,
                "metric": "throughput_speedup_pct",
                "n": n,
                "mean": mean,
                "sd": sd,
                "ci95": ci95,
            }
        )
    return aggregate_rows, speedup_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Run a small end-to-end PyTorch controller matrix.")
    p.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    p.add_argument("--controllers", default="static,headroom,safe_greedy")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out-dir", type=Path, default=Path("out/matrix"))
    p.add_argument("--baseline", default="static")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--budget-mb", type=float, default=None)
    p.add_argument("--rl-checkpoint", type=Path, default=None)
    p.add_argument("--rl-action-profile", choices=("all", "fast_only"), default=None)
    p.add_argument("--rl-device", type=str, default=None)
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_tiny.py"
    if args.config.is_absolute() or args.config.exists():
        config = args.config
    else:
        config = script_dir / args.config
    config = config.resolve()
    rl_checkpoint = args.rl_checkpoint
    if rl_checkpoint is not None:
        if rl_checkpoint.is_absolute() or rl_checkpoint.exists():
            rl_checkpoint = rl_checkpoint.resolve()
        else:
            rl_checkpoint = (script_dir / rl_checkpoint).resolve()
        if not rl_checkpoint.exists():
            raise SystemExit(f"RL checkpoint not found: {rl_checkpoint}")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for controller in _parse_csv(args.controllers):
        for seed in _parse_int_csv(args.seeds):
            telemetry = out_dir / f"{controller}__seed{seed}.csv"
            cmd = [
                sys.executable,
                str(train_script),
                "--config",
                str(config),
                "--controller",
                controller,
                "--output",
                str(telemetry),
                "--seed",
                str(seed),
            ]
            if controller == "rl":
                if rl_checkpoint is None:
                    raise SystemExit("--controllers includes rl, so --rl-checkpoint is required")
                cmd.extend(["--rl-checkpoint", str(rl_checkpoint)])
                if args.rl_action_profile is not None:
                    cmd.extend(["--rl-action-profile", str(args.rl_action_profile)])
                if args.rl_device is not None:
                    cmd.extend(["--rl-device", str(args.rl_device)])
            if args.steps is not None:
                cmd.extend(["--steps", str(int(args.steps))])
            if args.device is not None:
                cmd.extend(["--device", str(args.device)])
            if args.budget_mb is not None:
                cmd.extend(["--budget-mb", str(float(args.budget_mb))])
            subprocess.run(cmd, cwd=str(script_dir), check=True)
            row = _summarize_telemetry(telemetry)
            row["seed"] = seed
            row["config"] = str(config)
            summary_rows.append(row)

    summary_path = out_dir / "summary.csv"
    aggregate_path = out_dir / "aggregate_ci.csv"
    speedup_path = out_dir / "speedup_ci.csv"
    _write_csv(summary_path, summary_rows)
    aggregate_rows, speedup_rows = _aggregate(summary_rows, baseline=args.baseline)
    _write_csv(aggregate_path, aggregate_rows)
    _write_csv(speedup_path, speedup_rows)

    print(f"wrote {summary_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {speedup_path}")


if __name__ == "__main__":
    main()
