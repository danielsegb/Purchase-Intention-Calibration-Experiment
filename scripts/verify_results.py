"""Verify experiment outputs against baseline reference results.

Compares summary metrics, evaluates core experimental conclusions, and
displays benchmark tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs"
REFERENCE = ROOT / "reference_results"

KEYS = ["Model", "FeatureCondition", "Calibration"]
METRICS = ["Brier", "LogLoss", "ECE", "ROC_AUC", "PR_AUC", "F1"]


def canonical(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sort and reset index of dataframe by grouping keys."""
    return df.sort_values(keys).reset_index(drop=True)


def max_abs_differences(generated: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Compute maximum absolute difference per metric between two dataframes."""
    out = {}
    for metric in METRICS:
        out[metric] = float(
            np.max(
                np.abs(
                    generated[metric].to_numpy(dtype=float)
                    - reference[metric].to_numpy(dtype=float)
                )
            )
        )
    return out


def main(output_dir: Path, strict: bool, atol: float) -> None:
    """Run data integrity and metric verification on generated outputs."""
    generated_integrity = json.loads(
        (output_dir / "data_integrity_checks.json").read_text(encoding="utf-8")
    )
    reference_integrity = json.loads(
        (REFERENCE / "data_integrity_checks.json").read_text(encoding="utf-8")
    )
    if generated_integrity != reference_integrity:
        raise SystemExit("FAIL: data-integrity checks differ from the reference run.")

    generated = canonical(pd.read_csv(output_dir / "summary_mean_sd_ci.csv"), KEYS)
    reference = canonical(pd.read_csv(REFERENCE / "summary_mean_sd_ci.csv"), KEYS)

    if generated[KEYS].to_dict("records") != reference[KEYS].to_dict("records"):
        raise SystemExit("FAIL: result groups differ from the reference run.")

    diffs = max_abs_differences(generated, reference)

    print("Reference comparison (maximum absolute differences):")
    for metric, value in diffs.items():
        print(f"  {metric:8s} {value:.12g}")

    # Logistic Regression should be numerically identical across platforms.
    lr_generated = generated[generated["Model"] == "Logistic Regression"].reset_index(drop=True)
    lr_reference = reference[reference["Model"] == "Logistic Regression"].reset_index(drop=True)
    lr_diffs = max_abs_differences(lr_generated, lr_reference)
    worst_lr = max(lr_diffs.values())
    if worst_lr > atol:
        raise SystemExit(
            f"FAIL: Logistic Regression differs from reference beyond atol={atol}; "
            f"maximum difference={worst_lr:.12g}."
        )

    if strict:
        worst = max(diffs.values())
        if worst > atol:
            raise SystemExit(
                f"STRICT FAIL: at least one metric differs from reference beyond atol={atol}; "
                f"maximum difference={worst:.12g}. "
                "Strict mode is intended for identical compiler/hardware environments."
            )
        print("STRICT PASS: results match the reference run within tolerance.")
        return

    # Core experimental conclusion checks
    def value(model: str, feature: str, calibration: str, metric: str) -> float:
        row = generated[
            (generated["Model"] == model)
            & (generated["FeatureCondition"] == feature)
            & (generated["Calibration"] == calibration)
        ]
        if len(row) != 1:
            raise SystemExit(f"FAIL: missing or duplicated result row for {model}, {feature}, {calibration}.")
        return float(row.iloc[0][metric])

    checks = {
        "XGBoost has lower all-predictor uncalibrated Brier than Logistic Regression":
            value("XGBoost", "All predictors", "Uncalibrated", "Brier")
            < value("Logistic Regression", "All predictors", "Uncalibrated", "Brier"),
        "XGBoost has higher all-predictor uncalibrated PR-AUC than Logistic Regression":
            value("XGBoost", "All predictors", "Uncalibrated", "PR_AUC")
            > value("Logistic Regression", "All predictors", "Uncalibrated", "PR_AUC"),
        "Isotonic calibration improves Logistic Regression Brier score":
            value("Logistic Regression", "All predictors", "Isotonic", "Brier")
            < value("Logistic Regression", "All predictors", "Uncalibrated", "Brier"),
        "Platt scaling does not improve XGBoost Brier score":
            value("XGBoost", "All predictors", "Platt", "Brier")
            > value("XGBoost", "All predictors", "Uncalibrated", "Brier"),
        "Removing PageValues worsens Logistic Regression Brier score":
            value("Logistic Regression", "Without PageValues", "Uncalibrated", "Brier")
            > value("Logistic Regression", "All predictors", "Uncalibrated", "Brier"),
        "Removing PageValues worsens XGBoost Brier score":
            value("XGBoost", "Without PageValues", "Uncalibrated", "Brier")
            > value("XGBoost", "All predictors", "Uncalibrated", "Brier"),
        "Removing PageValues reduces Logistic Regression PR-AUC":
            value("Logistic Regression", "Without PageValues", "Uncalibrated", "PR_AUC")
            < value("Logistic Regression", "All predictors", "Uncalibrated", "PR_AUC"),
        "Removing PageValues reduces XGBoost PR-AUC":
            value("XGBoost", "Without PageValues", "Uncalibrated", "PR_AUC")
            < value("XGBoost", "All predictors", "Uncalibrated", "PR_AUC"),
    }

    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    if failed:
        raise SystemExit("FAIL: one or more core experimental conclusions were not reproduced.")

    print(
        "\nPORTABLE PASS: dataset integrity, Logistic Regression results, and core "
        "experimental conclusions verified successfully."
    )

    print("\n" + "=" * 95)
    print("BENCHMARK RESULTS (Mean +/- SD rounded to 4 decimals)")
    print("=" * 95)

    print("\n--- Table 1: Performance with all predictors across ten repetitions (Mean +/- SD) ---")
    t1_rows = generated[generated["FeatureCondition"] == "All predictors"]
    print(f"{'Model':<20} {'Calibration':<13} {'Brier':<17} {'Log loss':<17} {'ECE':<17} {'ROC-AUC':<17} {'PR-AUC':<17} {'F1':<17}")
    print("-" * 135)
    for _, r in t1_rows.iterrows():
        brier = f"{r['Brier']:.4f} +/- {r['Brier_SD']:.4f}"
        ll = f"{r['LogLoss']:.4f} +/- {r['LogLoss_SD']:.4f}"
        ece = f"{r['ECE']:.4f} +/- {r['ECE_SD']:.4f}"
        roc = f"{r['ROC_AUC']:.4f} +/- {r['ROC_AUC_SD']:.4f}"
        pr = f"{r['PR_AUC']:.4f} +/- {r['PR_AUC_SD']:.4f}"
        f1 = f"{r['F1']:.4f} +/- {r['F1_SD']:.4f}"
        print(f"{r['Model']:<20} {r['Calibration']:<13} {brier:<17} {ll:<17} {ece:<17} {roc:<17} {pr:<17} {f1:<17}")

    print("\n--- Table 2: Sensitivity to PageValues for uncalibrated models (Mean +/- SD) ---")
    t2_rows = generated[generated["Calibration"] == "Uncalibrated"]
    print(f"{'Model':<20} {'FeatureCondition':<20} {'Brier':<17} {'Log loss':<17} {'ECE':<17} {'ROC-AUC':<17} {'PR-AUC':<17} {'F1':<17}")
    print("-" * 142)
    for _, r in t2_rows.iterrows():
        brier = f"{r['Brier']:.4f} +/- {r['Brier_SD']:.4f}"
        ll = f"{r['LogLoss']:.4f} +/- {r['LogLoss_SD']:.4f}"
        ece = f"{r['ECE']:.4f} +/- {r['ECE_SD']:.4f}"
        roc = f"{r['ROC_AUC']:.4f} +/- {r['ROC_AUC_SD']:.4f}"
        pr = f"{r['PR_AUC']:.4f} +/- {r['PR_AUC_SD']:.4f}"
        f1 = f"{r['F1']:.4f} +/- {r['F1_SD']:.4f}"
        print(f"{r['Model']:<20} {r['FeatureCondition']:<20} {brier:<17} {ll:<17} {ece:<17} {roc:<17} {pr:<17} {f1:<17}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benchmark results against reference baseline.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Path to generated outputs directory.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require every summary metric to match the reference within --atol.",
    )
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance for strict check.")
    args = parser.parse_args()
    main(args.output_dir, args.strict, args.atol)
