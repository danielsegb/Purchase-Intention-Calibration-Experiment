"""Cross-platform script to reproduce the complete purchase-intention calibration experiment.

Runs the data download, experiment execution, and result verification in sequence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str], description: str) -> None:
    print(f"\n{'='*70}\n>> {description}\n   Command: {' '.join(cmd)}\n{'='*70}\n", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"\nERROR: Step failed with return code {res.returncode}: {description}", file=sys.stderr)
        sys.exit(res.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all steps to reproduce the paper results: download data, run models, verify outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to write experiment outputs (default: outputs)",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip strict Python/package version check (useful if testing with minor variations).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers for hyperparameter tuning (default: 1 for exact sequential reproduction; use -1 for all cores).",
    )
    args = parser.parse_args()

    py = sys.executable

    # 1. Download data
    run_step([py, str(ROOT / "scripts" / "download_data.py")], "Step 1: Download & verify dataset from UCI")

    # 2. Run experiment
    exp_cmd = [
        py,
        str(ROOT / "src" / "run_experiment.py"),
        str(ROOT / "data" / "online_shoppers_intention.csv"),
        "--output-dir",
        args.output_dir,
        "--n-jobs",
        str(args.n_jobs),
    ]
    if args.skip_env_check:
        exp_cmd.append("--skip-env-check")
    run_step(exp_cmd, "Step 2: Run 10-seed experiment (Logistic Regression & XGBoost)")

    # 3. Verify results
    run_step(
        [py, str(ROOT / "scripts" / "verify_results.py"), "--output-dir", args.output_dir],
        "Step 3: Verify results against reference outputs and paper tables",
    )

    print("\n" + "="*70)
    print("SUCCESS: Full reproduction workflow finished successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
