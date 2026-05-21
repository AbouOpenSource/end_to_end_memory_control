from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ValidationContractTest(unittest.TestCase):
    def test_train_tiny_writes_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "telemetry.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train_tiny.py"),
                    "--config",
                    str(ROOT / "configs" / "smoke.json"),
                    "--controller",
                    "safe_greedy",
                    "--steps",
                    "2",
                    "--output",
                    str(out),
                ],
                cwd=str(ROOT),
                check=True,
            )
            with out.open("r", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        for column in (
            "dataset_name",
            "precision",
            "memory_source",
            "effective_samples_per_s",
            "next_micro_batch",
            "next_grad_accum_steps",
        ):
            self.assertIn(column, rows[0])
        self.assertEqual(rows[-1]["dataset_name"], "synthetic")
        self.assertEqual(rows[-1]["memory_source"], "proxy")

    def test_matrix_and_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "cpu_contract"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_matrix.py"),
                    "--config",
                    str(ROOT / "configs" / "smoke.json"),
                    "--controllers",
                    "static,headroom",
                    "--seeds",
                    "0,1",
                    "--steps",
                    "2",
                    "--baseline",
                    "static",
                    "--out-dir",
                    str(case_dir),
                ],
                cwd=str(ROOT),
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "summarize_validation.py"),
                    str(root),
                    "--no-plots",
                    "--strict",
                ],
                cwd=str(ROOT),
                check=True,
            )

            with (case_dir / "summary.csv").open("r", newline="") as f:
                summary_rows = list(csv.DictReader(f))
            report = (root / "validation_report.md").read_text()

        self.assertEqual(len(summary_rows), 4)
        self.assertIn("cpu_contract", report)
        self.assertIn("PASS", report)


if __name__ == "__main__":
    unittest.main()
