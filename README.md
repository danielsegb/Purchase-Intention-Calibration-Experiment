# Online Purchase Intention: Probability Calibration & Sensitivity Benchmark

A benchmark experiment evaluating probability calibration and feature dependence in online purchase-intention prediction using the UCI Online Shoppers Purchasing Intention Dataset.

This project compares **Logistic Regression** and **XGBoost** across:
- **Probability accuracy & calibration**: Brier score, log loss, and Expected Calibration Error (ECE);
- **Post-hoc calibration**: Platt scaling (sigmoid) and isotonic regression;
- **Discrimination & classification**: ROC-AUC, PR-AUC (average precision), and F1-score at a 0.50 decision threshold;
- **Feature dependence**: Controlled sensitivity analysis evaluating model performance with and without `PageValues`.

---

## Quickstart

This repository uses a standard Python workflow with a virtual environment and pinned dependencies in `requirements.txt`.

### 1. Create and Activate Virtual Environment

The reference environment uses **Python 3.13** (tested with 3.13.5 and 3.13.x).

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Pinned versions:
- `numpy==2.3.5`
- `pandas==2.2.3`
- `scipy==1.17.0`
- `scikit-learn==1.8.0`
- `xgboost==3.1.3`
- `matplotlib==3.10.8`

### 3. Run the Benchmark

Run the automated pipeline (download dataset, train 10-seed cross-validated models, and verify outputs):

```bash
python scripts/run_all.py
```

*Tip: By default, hyperparameter tuning runs with `n_jobs=1` for deterministic single-worker scheduling. To accelerate tuning across all CPU cores on your machine, pass:*
```bash
python scripts/run_all.py --n-jobs -1
```

---

## Step-by-Step Execution

### Step 1: Download & Verify Dataset
```bash
python scripts/download_data.py
```
Downloads the official UCI archive and extracts `data/online_shoppers_intention.csv`. Validates data integrity against expected SHA-256:
`b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

### Step 2: Run the Experiment
```bash
python src/run_experiment.py data/online_shoppers_intention.csv --output-dir outputs
```
Options:
- `--n-jobs <int>`: Number of parallel workers for hyperparameter tuning (`1` default, `-1` for all CPU cores).
- `--skip-env-check`: Proceed even if Python or package versions differ from the reference environment.

### Step 3: Verify Results Against Reference Baseline
```bash
python scripts/verify_results.py --output-dir outputs
```
This checks:
1. Dataset integrity (12,330 original rows, 125 duplicates removed, 12,205 sessions retained, 15.63% positive rate).
2. Numerical consistency of Logistic Regression and XGBoost metrics against reference baseline.
3. Core experimental conclusions (XGBoost vs Logistic Regression discrimination, isotonic calibration effect, PageValues sensitivity).
4. Formatted summary tables for all-predictor evaluation and feature ablation.

---

## Experimental Protocol

- **Dataset**: UCI Online Shoppers Purchasing Intention (ID: 468, DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)).
- **Cleaning**: 125 exact duplicate rows removed before splitting, yielding 12,205 unique sessions (10,297 non-purchase, 1,908 purchase).
- **Partitions**: 10 repeated stratified splits (seeds 1 to 10), each allocating 60% training, 20% calibration, and 20% untouched test.
- **Model Tuning**: 5-fold stratified cross-validation on the training partition optimizing mean log loss.
- **Classifiers**:
  - **Logistic Regression**: L2 regularisation, $C \in \{0.01, 0.1, 1.0, 10.0\}$. Numerical predictors standardized using training parameters.
  - **XGBoost**: `hist` tree method, tuning `max_depth` (3, 5), `learning_rate` (0.05, 0.10), `n_estimators` (200, 400), `subsample` (0.8, 1.0), and `colsample_bytree` (0.8, 1.0).
- **Calibration**: Uncalibrated, Platt scaling (`sigmoid`), and Isotonic regression fitted exclusively on the dedicated calibration set.
- **Evaluation**: Brier score, log loss, expected calibration error (10 equal-width bins), ROC-AUC, PR-AUC, and F1-score at 0.50 threshold.

---

## Output Artifacts

Execution writes the following files to `outputs/`:

| File | Description |
| :--- | :--- |
| `summary_mean_sd_ci.csv` | Mean, standard deviation, and 95% Student-t CI across 10 repetitions |
| `paired_brier_contrasts.csv` | Paired differences and 95% CIs by seed across models and calibration conditions |
| `reliability_logistic_regression.png` | 10-bin reliability diagram for Logistic Regression (Uncalibrated, Platt, Isotonic) |
| `reliability_xgboost.png` | 10-bin reliability diagram for XGBoost (Uncalibrated, Platt, Isotonic) |
| `data_integrity_checks.json` | Sample counts, duplicate records, and target distribution checks |
| `tuning_results.csv` | Selected hyperparameters and cross-validation log loss for each seed |
| `tuning_parameter_frequencies.csv` | Frequency distribution of selected hyperparameters |
| `test_metrics_by_seed.csv` | Evaluation metrics per seed, model, feature condition, and calibration method |
| `test_predictions_by_seed.csv` | Raw test probabilities and observed binary outcomes |
| `software_versions.json` | Execution runtime, hardware platform, and package version metadata |

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── scripts/
│   ├── download_data.py
│   ├── run_all.py
│   └── verify_results.py
├── src/
│   └── run_experiment.py
├── outputs/
└── reference_results/
```
