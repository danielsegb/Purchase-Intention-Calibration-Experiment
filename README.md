# Calibration and Feature Dependence in Online Purchase Intention Prediction

Reproducibility repository for the manuscript:

**Calibration and Feature Dependence in Online Purchase Intention Prediction: Logistic Regression versus XGBoost on the UCI Online Shoppers Dataset**

Authors: Daniel Segboru Emmanuel, Ikiomoye Douglas Emmanuel, Princess Enyagam, and Ogechi Nelson.

## What this repository reproduces

The code reproduces the complete computational experiment reported in the manuscript:

- removal of 125 exact duplicate rows before partitioning;
- 12,205-session analytical sample;
- ten fixed random seeds (`1` to `10`);
- stratified 60% training / 20% calibration / 20% test partitions;
- Logistic Regression and XGBoost;
- training-only five-fold hyperparameter selection by mean log loss;
- uncalibrated probabilities, Platt scaling, and isotonic regression;
- all-predictor and `PageValues`-excluded conditions;
- Brier score, log loss, 10-bin ECE, ROC-AUC, PR-AUC/average precision, and positive-class F1 at 0.50;
- mean, sample SD, and 95% Student t intervals across the ten repetitions;
- seed-paired Brier-score contrasts;
- pooled all-predictor reliability diagrams.

The retained outputs used in preparing the manuscript are included under `reference_results/` so a fresh execution can be checked programmatically.

## Dataset

The experiment uses the **Online Shoppers Purchasing Intention Dataset**, UCI dataset ID 468.

- UCI record: https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset
- DOI: https://doi.org/10.24432/C5F88Q
- Creators: C. Sakar and Yomi Kastro
- Licence: CC BY 4.0

The repository does not need to store the raw CSV. The download helper retrieves the official UCI archive and verifies the exact CSV by SHA-256.

## Exact software environment

The reported analysis used:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- SciPy 1.17.0
- scikit-learn 1.8.0
- XGBoost 3.1.3
- Matplotlib 3.10.8

Exact Python-package versions are pinned in `requirements.txt` and `environment.yml`.

## Reproduce the experiment

### Option A: Python virtual environment

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download and verify the dataset:

```bash
python scripts/download_data.py
```

Run the full experiment:

```bash
python src/run_experiment.py data/online_shoppers_intention.csv --output-dir outputs
```

Verify the fresh run against the retained manuscript results:

```bash
python scripts/verify_results.py --output-dir outputs
```

A successful verification ends with:

```text
PASS: fresh results reproduce the retained manuscript analysis within tolerance.
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate purchase-intention-calibration
python scripts/download_data.py
python src/run_experiment.py data/online_shoppers_intention.csv --output-dir outputs
python scripts/verify_results.py --output-dir outputs
```

## Expected headline results

For the **all-predictor** condition, the retained analysis reports:

| Model | Calibration | Brier | Log loss | ECE | ROC-AUC | PR-AUC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Uncalibrated | 0.0868 | 0.2978 | 0.0416 | 0.8919 | 0.6376 | 0.4987 |
| Logistic Regression | Platt | 0.0866 | 0.2981 | 0.0397 | 0.8919 | 0.6376 | 0.5014 |
| Logistic Regression | Isotonic | 0.0838 | 0.3102 | 0.0162 | 0.8905 | 0.6132 | 0.5919 |
| XGBoost | Uncalibrated | 0.0696 | 0.2271 | 0.0145 | 0.9332 | 0.7472 | 0.6539 |
| XGBoost | Platt | 0.0720 | 0.2446 | 0.0264 | 0.9332 | 0.7472 | 0.6461 |
| XGBoost | Isotonic | 0.0701 | 0.2425 | 0.0153 | 0.9316 | 0.7217 | 0.6582 |

For the **uncalibrated without-PageValues** condition:

| Model | Brier | Log loss | ECE | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1196 | 0.3829 | 0.0128 | 0.7416 | 0.3242 | 0.0468 |
| XGBoost | 0.1152 | 0.3665 | 0.0141 | 0.7740 | 0.3630 | 0.1148 |

Values above are rounded for presentation. Paired differences in the manuscript are calculated from the unrounded seed-level values stored in `test_metrics_by_seed.csv`.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── archive/
│   └── original_analysis_script.py
├── data/
│   └── README.md
├── docs/
│   ├── METHODS.md
│   ├── OUTPUTS.md
│   └── PROVENANCE.md
├── scripts/
│   ├── download_data.py
│   └── verify_results.py
├── src/
│   └── run_experiment.py
├── outputs/
└── reference_results/
    ├── data_integrity_checks.json
    ├── test_metrics_by_seed.csv
    ├── summary_mean_sd_ci.csv
    ├── tuning_results.csv
    ├── tuning_parameter_frequencies.csv
    ├── paired_brier_contrasts.csv
    ├── pooled_reliability_data.csv
    └── figures/
```

## Important interpretation note

The ten repetitions are repeated stratified holdout partitions drawn from the same dataset. Their confidence intervals summarise **split-to-split variability** and should not be interpreted as independent population-level inferential intervals or external validation.

The benchmark analysis uses complete session-level records. The manuscript therefore does not claim that the evaluated full-predictor configuration has been validated as an early-session real-time intervention system.

## Code availability statement for the manuscript

> **Code availability:** The complete analysis code, fixed random seeds, software environment, retained reference outputs, and instructions required to reproduce the experiments are publicly available at: https://github.com/danielsegb/purchase-intention-calibration-reproducibility

## Citation

Please cite the associated manuscript and the UCI dataset. GitHub can also generate a software citation from `CITATION.cff`.