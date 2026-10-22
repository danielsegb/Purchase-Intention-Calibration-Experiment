from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs"
REFERENCE = ROOT / "reference_results"

KEYS = ["Model", "FeatureCondition", "Calibration"]
METRIC_COLS = [
    "Brier", "Brier_SD", "Brier_CI_low", "Brier_CI_high",
    "LogLoss", "LogLoss_SD", "LogLoss_CI_low", "LogLoss_CI_high",
    "ECE", "ECE_SD", "ECE_CI_low", "ECE_CI_high",
    "ROC_AUC", "ROC_AUC_SD", "ROC_AUC_CI_low", "ROC_AUC_CI_high",
    "PR_AUC", "PR_AUC_SD", "PR_AUC_CI_low", "PR_AUC_CI_high",
    "F1", "F1_SD", "F1_CI_low", "F1_CI_high",
]


def canonical(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(keys).reset_index(drop=True)


def main(output_dir: Path, atol: float) -> None:
    generated = canonical(pd.read_csv(output_dir / "summary_mean_sd_ci.csv"), KEYS)
    reference = canonical(pd.read_csv(REFERENCE / "summary_mean_sd_ci.csv"), KEYS)

    if generated[KEYS].to_dict("records") != reference[KEYS].to_dict("records"):
        raise SystemExit("FAIL: result groups differ from the reference analysis.")

    diffs = []
    for col in METRIC_COLS:
        max_abs = float(np.max(np.abs(generated[col].to_numpy() - reference[col].to_numpy())))
        diffs.append((col, max_abs))

    worst_col, worst = max(diffs, key=lambda x: x[1])
    print(f"Maximum absolute metric difference: {worst:.12g} ({worst_col})")
    if worst > atol:
        raise SystemExit(f"FAIL: fresh run differs from reference results beyond atol={atol}.")

    g_tune = canonical(pd.read_csv(output_dir / "tuning_results.csv"), ["Seed", "Model", "FeatureCondition"])
    r_tune = canonical(pd.read_csv(REFERENCE / "tuning_results.csv"), ["Seed", "Model", "FeatureCondition"])
    if not g_tune[["Seed", "Model", "FeatureCondition", "BestParams"]].equals(
        r_tune[["Seed", "Model", "FeatureCondition", "BestParams"]]
    ):
        raise SystemExit("FAIL: at least one selected hyperparameter configuration differs.")

    print("PASS: fresh results reproduce the reference experiment run within tolerance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    main(args.output_dir, args.atol)
