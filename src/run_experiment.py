from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEEDS = list(range(1, 11))
TARGET = "Revenue"
FEATURE_CONDITIONS = {
    "All predictors": None,
    "Without PageValues": "PageValues",
}
LR_GRID = {"model__C": [0.01, 0.1, 1.0, 10.0]}
XGB_GRID = {
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.10],
    "model__n_estimators": [200, 400],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}
METRICS = ["Brier", "LogLoss", "ECE", "ROC_AUC", "PR_AUC", "F1"]


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error using equal-width bins over [0, 1]."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - p[mask].mean())
    return float(ece)


def metric_row(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    eps = np.finfo(float).eps
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return {
        "Brier": brier_score_loss(y_true, p),
        "LogLoss": log_loss(y_true, p, labels=[0, 1]),
        "ECE": expected_calibration_error(y_true, p, n_bins=10),
        "ROC_AUC": roc_auc_score(y_true, p),
        "PR_AUC": average_precision_score(y_true, p),
        "F1": f1_score(y_true, (p >= 0.50).astype(int), pos_label=1),
    }


def infer_column_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    X = df.drop(columns=[target])
    categorical = [
        c
        for c in [
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
        ]
        if c in X.columns
    ]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_lr_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def build_xgb_pipeline(numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def fit_tuned_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric: list[str],
    categorical: list[str],
    seed: int,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if model_name == "Logistic Regression":
        pipe = build_lr_pipeline(numeric, categorical)
        grid = LR_GRID
    elif model_name == "XGBoost":
        pipe = build_xgb_pipeline(numeric, categorical, seed)
        grid = XGB_GRID
    else:
        raise ValueError(model_name)

    search = GridSearchCV(
        pipe,
        param_grid=grid,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, -search.best_score_


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["Model", "FeatureCondition", "Calibration"]
    tcrit = stats.t.ppf(0.975, df=9)
    for keys, g in results.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        for metric in METRICS:
            vals = g[metric].to_numpy(float)
            mean = vals.mean()
            sd = vals.std(ddof=1)
            half = tcrit * sd / np.sqrt(len(vals))
            row[metric] = mean
            row[f"{metric}_SD"] = sd
            row[f"{metric}_CI_low"] = mean - half
            row[f"{metric}_CI_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sd = values.std(ddof=1)
    half = stats.t.ppf(0.975, df=len(values) - 1) * sd / np.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def paired_brier_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # Calibration contrasts: positive means calibration reduced Brier score.
    for (model, feature_condition), g in results.groupby(["Model", "FeatureCondition"], sort=False):
        wide = g.pivot(index="Seed", columns="Calibration", values="Brier")
        for cal in ["Platt", "Isotonic"]:
            d = (wide["Uncalibrated"] - wide[cal]).dropna().to_numpy(float)
            mean, sd, lo, hi = paired_summary(d)
            rows.append({
                "Definition": f"{model}, {feature_condition}: Uncalibrated - {cal}",
                "Metric": "Brier",
                "MeanDifference": mean,
                "SD": sd,
                "CI_low": lo,
                "CI_high": hi,
            })

    # PageValues contrasts: positive means removal worsened Brier score.
    for (model, calibration), g in results.groupby(["Model", "Calibration"], sort=False):
        wide = g.pivot(index="Seed", columns="FeatureCondition", values="Brier")
        d = (wide["Without PageValues"] - wide["All predictors"]).dropna().to_numpy(float)
        mean, sd, lo, hi = paired_summary(d)
        rows.append({
            "Definition": f"{model}, {calibration}: Without PageValues - All predictors",
            "Metric": "Brier",
            "MeanDifference": mean,
            "SD": sd,
            "CI_low": lo,
            "CI_high": hi,
        })

    # Cross-model contrasts: positive means LR has the larger Brier score.
    for (feature_condition, calibration), g in results.groupby(["FeatureCondition", "Calibration"], sort=False):
        wide = g.pivot(index="Seed", columns="Model", values="Brier")
        d = (wide["Logistic Regression"] - wide["XGBoost"]).dropna().to_numpy(float)
        mean, sd, lo, hi = paired_summary(d)
        rows.append({
            "Definition": f"{feature_condition}, {calibration}: Logistic Regression - XGBoost",
            "Metric": "Brier",
            "MeanDifference": mean,
            "SD": sd,
            "CI_low": lo,
            "CI_high": hi,
        })

    return pd.DataFrame(rows)


def pooled_reliability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    edges = np.linspace(0.0, 1.0, 11)
    subset = predictions[predictions["FeatureCondition"] == "All predictors"].copy()
    # Match scikit-learn calibration_curve(strategy="uniform"): values exactly
    # on an internal bin edge are assigned to the lower bin. This matters for
    # isotonic calibration, which can produce repeated probabilities at edges.
    subset["Bin"] = np.searchsorted(edges[1:-1], subset["Probability"].to_numpy())
    for (model, calibration, bin_id), g in subset.groupby(["Model", "Calibration", "Bin"], sort=True):
        rows.append({
            "Model": model,
            "Calibration": calibration,
            "Bin": int(bin_id),
            "MeanPredicted": g["Probability"].mean(),
            "ObservedFrequency": g["Observed"].mean(),
            "Count": int(len(g)),
        })
    return pd.DataFrame(rows)


def make_reliability_figure(pooled: pd.DataFrame, model: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.7))
    for calibration in ["Uncalibrated", "Platt", "Isotonic"]:
        g = pooled[(pooled["Model"] == model) & (pooled["Calibration"] == calibration)].sort_values("Bin")
        ax.plot(g["MeanPredicted"], g["ObservedFrequency"], marker="o", label=calibration)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed purchase frequency")
    ax.set_title(f"Reliability diagram, {model}, all predictors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(csv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df0 = pd.read_csv(csv_path)

    checks = {
        "original_rows": int(df0.shape[0]),
        "original_columns": int(df0.shape[1]),
        "exact_duplicate_rows": int(df0.duplicated().sum()),
        "missing_cells": int(df0.isna().sum().sum()),
    }
    df = df0.drop_duplicates().reset_index(drop=True)
    checks["rows_after_duplicate_removal"] = int(df.shape[0])

    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' not found. Columns: {list(df.columns)}")

    if pd.api.types.is_bool_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].astype(int)
    elif not pd.api.types.is_numeric_dtype(df[TARGET]):
        mapping = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, "1": 1, "0": 0}
        df[TARGET] = df[TARGET].astype(str).map(mapping)
        if df[TARGET].isna().any():
            raise ValueError("Could not convert Revenue target to binary values.")
    else:
        df[TARGET] = df[TARGET].astype(int)

    checks["positive_class_count_after_dedup"] = int(df[TARGET].sum())
    checks["negative_class_count_after_dedup"] = int((1 - df[TARGET]).sum())
    checks["positive_class_rate_after_dedup"] = float(df[TARGET].mean())

    with (output_dir / "data_integrity_checks.json").open("w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    all_rows: list[dict] = []
    tuning_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for seed in SEEDS:
        print(f"Seed {seed}/10", flush=True)
        train_cal, test = train_test_split(
            df,
            test_size=0.20,
            stratify=df[TARGET],
            random_state=seed,
        )
        train, cal = train_test_split(
            train_cal,
            test_size=0.25,
            stratify=train_cal[TARGET],
            random_state=seed,
        )

        for feature_condition, excluded in FEATURE_CONDITIONS.items():
            feature_cols = [c for c in df.columns if c != TARGET and c != excluded]
            X_train, y_train = train[feature_cols], train[TARGET]
            X_cal, y_cal = cal[feature_cols], cal[TARGET]
            X_test, y_test = test[feature_cols], test[TARGET]

            numeric_all, categorical_all = infer_column_types(df[feature_cols + [TARGET]], TARGET)
            numeric = [c for c in numeric_all if c in feature_cols]
            categorical = [c for c in categorical_all if c in feature_cols]

            for model_name in ["Logistic Regression", "XGBoost"]:
                t0 = time.perf_counter()
                best, best_params, cv_logloss = fit_tuned_model(
                    model_name, X_train, y_train, numeric, categorical, seed
                )
                tune_seconds = time.perf_counter() - t0
                tuning_rows.append({
                    "Seed": seed,
                    "Model": model_name,
                    "FeatureCondition": feature_condition,
                    "BestParams": json.dumps(best_params, sort_keys=True),
                    "TrainingCV_LogLoss": cv_logloss,
                    "TuningSeconds": tune_seconds,
                })

                predictors = {"Uncalibrated": best}
                for label, method in [("Platt", "sigmoid"), ("Isotonic", "isotonic")]:
                    calibrated = CalibratedClassifierCV(FrozenEstimator(best), method=method)
                    calibrated.fit(X_cal, y_cal)
                    predictors[label] = calibrated

                for cal_label, estimator in predictors.items():
                    p = estimator.predict_proba(X_test)[:, 1]
                    row = {
                        "Seed": seed,
                        "Model": model_name,
                        "FeatureCondition": feature_condition,
                        "Calibration": cal_label,
                    }
                    row.update(metric_row(y_test.to_numpy(), p))
                    all_rows.append(row)

                    # Raw prediction storage enables exact pooled reliability reconstruction.
                    for row_index, observed, probability in zip(X_test.index, y_test.to_numpy(), p):
                        prediction_rows.append({
                            "Seed": seed,
                            "Model": model_name,
                            "FeatureCondition": feature_condition,
                            "Calibration": cal_label,
                            "RowIndex": int(row_index),
                            "Observed": int(observed),
                            "Probability": float(probability),
                        })

    results = pd.DataFrame(all_rows)
    tuning = pd.DataFrame(tuning_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarise(results)
    paired = paired_brier_contrasts(results)
    pooled = pooled_reliability(predictions)

    freq = (
        tuning.groupby(["Model", "FeatureCondition", "BestParams"], sort=False)
        .size()
        .reset_index(name="Count")
    )

    results.to_csv(output_dir / "test_metrics_by_seed.csv", index=False)
    tuning.to_csv(output_dir / "tuning_results.csv", index=False)
    freq.to_csv(output_dir / "tuning_parameter_frequencies.csv", index=False)
    summary.to_csv(output_dir / "summary_mean_sd_ci.csv", index=False)
    paired.to_csv(output_dir / "paired_brier_contrasts.csv", index=False)
    pooled.to_csv(output_dir / "pooled_reliability_data.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions_by_seed.csv", index=False)

    make_reliability_figure(
        pooled,
        "Logistic Regression",
        output_dir / "reliability_logistic_regression.png",
    )
    make_reliability_figure(
        pooled,
        "XGBoost",
        output_dir / "reliability_xgboost.png",
    )

    versions = {
        "Python": platform.python_version(),
        "pandas": pd.__version__,
        "NumPy": np.__version__,
        "SciPy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "XGBoost": xgboost.__version__,
        "Matplotlib": plt.matplotlib.__version__,
    }
    with (output_dir / "software_versions.json").open("w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)

    print("\nData checks:")
    print(json.dumps(checks, indent=2))
    print("\nSoftware versions:")
    print(json.dumps(versions, indent=2))
    print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the purchase-intention calibration experiment.")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path("data/online_shoppers_intention.csv"),
        help="Path to online_shoppers_intention.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    main(args.csv, args.output_dir)
