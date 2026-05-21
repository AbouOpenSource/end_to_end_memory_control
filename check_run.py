#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_run.py path/to/telemetry.csv")

    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"telemetry file not found: {path}")

    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"telemetry file has no rows: {path}")

    total_samples = sum(_as_float(row, "samples") for row in rows)
    step_times = [_as_float(row, "step_s") for row in rows]
    total_step_s = sum(step_times)
    peaks = [_as_float(row, "peak_mb") for row in rows]
    max_peak_mb = max(peaks)
    mean_peak_mb = sum(peaks) / max(len(peaks), 1)
    oom_count = sum(int(float(row.get("oom", "0") or 0)) for row in rows)
    knob_changes = sum(int(float(row.get("knob_changed", "0") or 0)) for row in rows)
    cuda_peak_reserved = max(_as_float(row, "cuda_peak_reserved_mb") for row in rows)
    final = rows[-1]

    throughput = total_samples / max(total_step_s, 1e-12)
    oom_rate = oom_count / max(len(rows), 1)

    print(f"rows={len(rows)}")
    if final.get("model_name"):
        print(
            "model="
            f"{final.get('model_name')} "
            f"dataset={final.get('dataset_name', 'unknown')} "
            f"precision={final.get('precision', 'unknown')} "
            f"params={final.get('model_parameters', 'unknown')} "
            f"input_shape={final.get('model_input_shape', 'unknown')}"
        )
    print(f"throughput_samples_per_s={throughput:.4f}")
    print(f"mean_step_s={sum(step_times) / max(len(step_times), 1):.6f}")
    print(f"p95_step_s={_percentile(step_times, 95):.6f}")
    print(f"mean_peak_mb={mean_peak_mb:.2f}")
    print(f"max_peak_mb={max_peak_mb:.2f}")
    if cuda_peak_reserved > 0:
        print(f"cuda_peak_reserved_mb={cuda_peak_reserved:.2f}")
    if final.get("memory_source"):
        print(f"memory_source={final.get('memory_source')}")
    print(f"oom_rate={oom_rate:.4f}")
    print(f"knob_changes={knob_changes}")
    print(
        "final_knobs="
        f"micro_batch={final.get('next_micro_batch', final.get('micro_batch'))} "
        f"grad_accum_steps={final.get('next_grad_accum_steps', final.get('grad_accum_steps'))}"
    )


if __name__ == "__main__":
    main()
