# Purchase Intention Calibration Experiment

Reproducible experiment comparing **Logistic Regression** and **XGBoost** for online purchase-intention prediction using the UCI Online Shoppers Purchasing Intention Dataset.

This study investigates:
- **Probability accuracy & calibration**: Brier score, log loss, and Expected Calibration Error (ECE);
- **Post-hoc calibration**: Platt scaling (sigmoid) and isotonic regression;
- **Discrimination & classification**: ROC-AUC, PR-AUC (average precision), and F1-score at a 0.50 threshold;
- **Feature dependence**: Sensitivity analysis evaluating model performance with and without `PageValues`.

---

## Quickstart (Reproduce in 3 Steps)

This repository is designed so that any technical person can reproduce the paper's results using standard Python without Docker or complex setup.

### 1. Create and Activate Virtual Environment

The paper experiments were conducted using **Python 3.13** (reference environment: Python 3.13.5). Any standard Python 3.13 interpreter (e.g., 3.13.0 to 3.13.x) is supported.

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

### 2. Install Pinned Dependencies

Install the exact package versions used in the paper:

```bash
pip install -r requirements.txt
```

Pinned dependencies in `requirements.txt`:
- `numpy==2.3.5`
- `pandas==2.2.3`
- `scipy==1.17.0`
- `scikit-learn==1.8.0`
- `xgboost==3.1.3`
- `matplotlib==3.10.8`

### 3. Run the Complete Experiment & Verification

You can run the entire automated pipeline (download dataset, train 10-seed repeated cross-validation models, generate outputs, and verify against paper tables) with a single command:

```bash
python scripts/run_all.py
```

*Tip: By default, hyperparameter tuning runs with `n_jobs=1` for strict sequential determinism. To speed up tuning across multiple CPU cores on your machine, pass:*
```bash
python scripts/run_all.py --n-jobs -1
```

---

## Step-by-Step Execution

If you prefer to run each step manually:

### Step 1: Download & Verify Dataset
```bash
python scripts/download_data.py
```
This downloads the official UCI archive and extracts `data/online_shoppers_intention.csv`. It verifies data integrity against the reported SHA-256 hash:
`b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

### Step 2: Run the Experiment
```bash
python src/run_experiment.py data/online_shoppers_intention.csv --output-dir outputs
```
Optional flags:
- `--n-jobs <int>`: Number of parallel workers for hyperparameter tuning (`1` default, `-1` for all CPU cores).
- `--skip-env-check`: Proceed even if Python or package versions differ from the reference environment.

### Step 3: Verify Results Against Paper Tables
```bash
python scripts/verify_results.py --output-dir outputs
```
This verifies:
1. Exact dataset integrity (12,330 original rows, 125 duplicates removed, 12,205 retained sessions, 15.63% positive rate).
2. Numerical replication of Logistic Regression and XGBoost metrics.
3. Core scientific conclusions of the paper.
4. Formatted side-by-side comparison tables against **Table 1** and **Table 2** of the paper.

---

## Experimental Design Summary

- **Dataset**: UCI Online Shoppers Purchasing Intention (ID: 468, DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)).
- **Cleaning**: 125 exact duplicate rows removed before splitting, yielding 12,205 sessions (10,297 non-purchase, 1,908 purchase).
- **Partitions**: 10 repeated stratified splits (seeds 1 to 10), each allocating 60% training, 20% calibration, and 20% untouched test.
- **Tuning**: 5-fold stratified cross-validation on training data optimizing mean log loss.
- **Models**:
  - **Logistic Regression**: L2 regularisation, $C \in \{0.01, 0.1, 1.0, 10.0\}$. Numerical predictors standardized using training parameters.
  - **XGBoost**: `hist` tree method, tuning `max_depth` (3, 5), `learning_rate` (0.05, 0.10), `n_estimators` (200, 400), `subsample` (0.8, 1.0), and `colsample_bytree` (0.8, 1.0).
- **Calibration**: Uncalibrated, Platt scaling (`sigmoid`), and Isotonic regression fitted exclusively on the dedicated calibration set.

---

## Output Files

The experiment generates the following artifacts in the `outputs/` folder:

| File | Description | Paper Reference |
| :--- | :--- | :--- |
| `summary_mean_sd_ci.csv` | Summary statistics (mean, SD, 95% CI) across 10 repetitions | **Table 1 & Table 2** |
| `paired_brier_contrasts.csv` | Paired seed-level differences and 95% CIs | **Sections 3.1–3.3** |
| `reliability_logistic_regression.png` | 10-bin reliability diagram for Logistic Regression | **Figure 2** |
| `reliability_xgboost.png` | 10-bin reliability diagram for XGBoost | **Figure 3** |
| `data_integrity_checks.json` | Sample size, class counts, and duplicate counts | **Section 2.2** |
| `tuning_results.csv` | Best hyperparameters and CV log loss for each seed | **Section 2.5** |
| `tuning_parameter_frequencies.csv` | Parameter selection distribution across repetitions | **Section 2.5** |
| `test_metrics_by_seed.csv` | Individual test metrics for every partition seed | **Section 3** |
| `test_predictions_by_seed.csv` | Raw test probabilities and observations | **Section 3** |

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   ├── METHODS.md
│   └── OUTPUTS.md
├── scripts/
│   ├── download_data.py
│   ├── run_all.py
│   └── verify_results.py
├── src/
│   └── run_experiment.py
├── outputs/
└── reference_results/
```
