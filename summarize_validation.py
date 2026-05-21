#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a Markdown report and optional plots from validation outputs.")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--strict", action="store_true", help="Exit non-zero when validation checks fail")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _metric_lookup(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["controller"], row["metric"]): row for row in rows if row.get("controller") and row.get("metric")}


def _case_dirs(out_dir: Path) -> list[Path]:
    return sorted(path for path in out_dir.iterdir() if path.is_dir() and (path / "summary.csv").exists())


def _collect_checks(case_name: str, summary_rows: list[dict[str, str]], aggregate_rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    if not summary_rows:
        return [f"{case_name}: missing or empty summary.csv"]
    if not aggregate_rows:
        issues.append(f"{case_name}: missing or empty aggregate_ci.csv")
    for row in summary_rows:
        controller = row.get("controller", "unknown")
        seed = row.get("seed", "unknown")
        throughput = _as_float(row, "throughput_samples_per_s")
        if not math.isfinite(throughput) or throughput <= 0:
            issues.append(f"{case_name}/{controller}/seed={seed}: invalid throughput")
        if row.get("memory_source") == "proxy" and row.get("dataset_name") != "synthetic":
            issues.append(f"{case_name}/{controller}/seed={seed}: real-dataset run used proxy memory")
    return issues


def _write_case_table(lines: list[str], case_dir: Path, summary_rows: list[dict[str, str]], aggregate_rows: list[dict[str, str]]) -> None:
    lookup = _metric_lookup(aggregate_rows)
    controllers = sorted({row.get("controller", "") for row in summary_rows if row.get("controller")})
    first = summary_rows[0] if summary_rows else {}
    lines.append(f"## {case_dir.name}")
    lines.append("")
    lines.append(
        f"- dataset: `{first.get('dataset_name', 'unknown')}`; "
        f"model: `{first.get('model_name', 'unknown')}`; "
        f"precision: `{first.get('precision', 'unknown')}`; "
        f"memory: `{first.get('memory_source', 'unknown')}`; "
        f"distributed: `{first.get('distributed', '0')}`; "
        f"world_size: `{first.get('world_size', '1')}`"
    )
    lines.append("")
    lines.append("| controller | n | throughput samples/s | max peak MB | OOM rate | knob changes |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for controller in controllers:
        thr = lookup.get((controller, "throughput_samples_per_s"), {})
        peak = lookup.get((controller, "max_peak_mb"), {})
        oom = lookup.get((controller, "oom_rate"), {})
        changes = lookup.get((controller, "knob_changes"), {})
        n = thr.get("n", "")
        thr_mean = _as_float(thr, "mean")
        thr_ci = _as_float(thr, "ci95")
        peak_mean = _as_float(peak, "mean")
        oom_mean = _as_float(oom, "mean")
        changes_mean = _as_float(changes, "mean")
        thr_text = _fmt(thr_mean, 2)
        if math.isfinite(thr_ci):
            thr_text = f"{thr_text} +/- {_fmt(thr_ci, 2)}"
        lines.append(
            f"| `{controller}` | {n} | {thr_text} | {_fmt(peak_mean, 2)} | "
            f"{_fmt(oom_mean, 4)} | {_fmt(changes_mean, 2)} |"
        )
    lines.append("")

    speedup_rows = _read_csv(case_dir / "speedup_ci.csv")
    if speedup_rows:
        lines.append("| controller | baseline | throughput speedup % | n |")
        lines.append("|---|---|---:|---:|")
        for row in speedup_rows:
            mean = _as_float(row, "mean")
            ci95 = _as_float(row, "ci95")
            value = _fmt(mean, 2)
            if math.isfinite(ci95):
                value = f"{value} +/- {_fmt(ci95, 2)}"
            lines.append(
                f"| `{row.get('controller', '')}` | `{row.get('baseline', '')}` | "
                f"{value} | {row.get('n', '')} |"
            )
        lines.append("")


def _plot_metric(case_dir: Path, aggregate_rows: list[dict[str, str]], metric: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in aggregate_rows if row.get("metric") == metric]
    if not rows:
        return
    labels = [row["controller"] for row in rows]
    means = [_as_float(row, "mean", 0.0) for row in rows]
    errors = [_as_float(row, "ci95", 0.0) for row in rows]
    errors = [0.0 if not math.isfinite(value) else value for value in errors]

    plot_dir = case_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(labels, means, yerr=errors, capsize=4, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"][: len(labels)])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{case_dir.name}: {metric}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{metric}.png", dpi=160)
    plt.close(fig)


def _write_plots(case_dir: Path, aggregate_rows: list[dict[str, str]]) -> None:
    try:
        _plot_metric(case_dir, aggregate_rows, "throughput_samples_per_s", "samples/s")
        _plot_metric(case_dir, aggregate_rows, "max_peak_mb", "MB")
        _plot_metric(case_dir, aggregate_rows, "oom_rate", "rate")
    except ImportError:
        print("matplotlib is not installed; skipping plots", file=sys.stderr)


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    metadata_path = out_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)

    lines: list[str] = ["# End-to-End Memory Control Validation", ""]
    if metadata:
        hardware = metadata.get("hardware", {})
        lines.append("## Hardware")
        lines.append("")
        for key in ("torch", "cuda_available", "cuda_version", "device_name", "total_memory_mb"):
            if key in hardware:
                lines.append(f"- {key}: `{hardware[key]}`")
        lines.append("")

    all_issues: list[str] = []
    for case_dir in _case_dirs(out_dir):
        summary_rows = _read_csv(case_dir / "summary.csv")
        aggregate_rows = _read_csv(case_dir / "aggregate_ci.csv")
        all_issues.extend(_collect_checks(case_dir.name, summary_rows, aggregate_rows))
        _write_case_table(lines, case_dir, summary_rows, aggregate_rows)
        if not args.no_plots:
            _write_plots(case_dir, aggregate_rows)

    lines.append("## Validation checks")
    lines.append("")
    if all_issues:
        for issue in all_issues:
            lines.append(f"- FAIL: {issue}")
    else:
        lines.append("- PASS: all summaries contain finite throughput and expected telemetry.")
    lines.append("")

    report_path = out_dir / "validation_report.md"
    with report_path.open("w") as f:
        f.write("\n".join(lines))
    print(f"wrote {report_path}")

    if args.strict and all_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
