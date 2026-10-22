from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs"
REFERENCE = ROOT / "reference_results"

SUMMARY_KEYS = ["Model", "FeatureCondition", "Calibration"]
SUMMARY_METRIC_COLS = [
    "Brier", "Brier_SD", "Brier_CI_low", "Brier_CI_high",
    "LogLoss", "LogLoss_SD", "LogLoss_CI_low", "LogLoss_CI_high",
    "ECE", "ECE_SD", "ECE_CI_low", "ECE_CI_high",
    "ROC_AUC", "ROC_AUC_SD", "ROC_AUC_CI_low", "ROC_AUC_CI_high",
    "PR_AUC", "PR_AUC_SD", "PR_AUC_CI_low", "PR_AUC_CI_high",
    "F1", "F1_SD", "F1_CI_low", "F1_CI_high",
]


def canonical(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(keys).reset_index(drop=True)


def assert_close_table(
    generated_path: Path,
    reference_path: Path,
    keys: list[str],
    numeric_columns: list[str],
    atol: float,
) -> None:
    generated = canonical(pd.read_csv(generated_path), keys)
    reference = canonical(pd.read_csv(reference_path), keys)

    if generated[keys].to_dict("records") != reference[keys].to_dict("records"):
        raise SystemExit(f"FAIL: grouping keys differ for {generated_path.name}.")

    for col in numeric_columns:
        max_abs = float(
            np.max(
                np.abs(
                    generated[col].to_numpy(dtype=float)
                    - reference[col].to_numpy(dtype=float)
                )
            )
        )
        if max_abs > atol:
            raise SystemExit(
                f"FAIL: {generated_path.name} differs in {col}; "
                f"maximum absolute difference={max_abs:.12g}, atol={atol}."
            )


def main(output_dir: Path, atol: float) -> None:
    # Data integrity checks.
    generated_integrity = json.loads(
        (output_dir / "data_integrity_checks.json").read_text(encoding="utf-8")
    )
    reference_integrity = json.loads(
        (REFERENCE / "data_integrity_checks.json").read_text(encoding="utf-8")
    )
    if generated_integrity != reference_integrity:
        raise SystemExit("FAIL: data-integrity checks differ from the reference run.")

    # Summary metrics.
    assert_close_table(
        output_dir / "summary_mean_sd_ci.csv",
        REFERENCE / "summary_mean_sd_ci.csv",
        SUMMARY_KEYS,
        SUMMARY_METRIC_COLS,
        atol,
    )

    # Seed-level metrics.
    seed_metric_cols = ["Brier", "LogLoss", "ECE", "ROC_AUC", "PR_AUC", "F1"]
    assert_close_table(
        output_dir / "test_metrics_by_seed.csv",
        REFERENCE / "test_metrics_by_seed.csv",
        ["Seed", "Model", "FeatureCondition", "Calibration"],
        seed_metric_cols,
        atol,
    )

    # Hyperparameter selections.
    tuning_keys = ["Seed", "Model", "FeatureCondition"]
    generated_tuning = canonical(
        pd.read_csv(output_dir / "tuning_results.csv"), tuning_keys
    )
    reference_tuning = canonical(
        pd.read_csv(REFERENCE / "tuning_results.csv"), tuning_keys
    )
    if generated_tuning[tuning_keys].to_dict("records") != reference_tuning[
        tuning_keys
    ].to_dict("records"):
        raise SystemExit("FAIL: tuning groups differ from the reference run.")
    if not generated_tuning["BestParams"].equals(reference_tuning["BestParams"]):
        raise SystemExit("FAIL: at least one selected hyperparameter configuration differs.")
    max_cv_diff = float(
        np.max(
            np.abs(
                generated_tuning["TrainingCV_LogLoss"].to_numpy(dtype=float)
                - reference_tuning["TrainingCV_LogLoss"].to_numpy(dtype=float)
            )
        )
    )
    if max_cv_diff > atol:
        raise SystemExit(
            f"FAIL: tuning CV log loss differs; maximum absolute difference="
            f"{max_cv_diff:.12g}, atol={atol}."
        )

    # Paired Brier-score contrasts.
    assert_close_table(
        output_dir / "paired_brier_contrasts.csv",
        REFERENCE / "paired_brier_contrasts.csv",
        ["Definition", "Metric"],
        ["MeanDifference", "SD", "CI_low", "CI_high"],
        atol,
    )

    # Reliability data.
    assert_close_table(
        output_dir / "pooled_reliability_data.csv",
        REFERENCE / "pooled_reliability_data.csv",
        ["Model", "Calibration", "Bin"],
        ["MeanPredicted", "ObservedFrequency", "Count"],
        atol,
    )

    print("PASS: fresh results reproduce the reference experiment run within tolerance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    main(args.output_dir, args.atol)
