# Purchase Intention Calibration Experiment

Reproducible experiment comparing **Logistic Regression** and **XGBoost** for online purchase-intention prediction using the UCI Online Shoppers Purchasing Intention Dataset.

The experiment evaluates:

- uncalibrated probabilities;
- Platt scaling;
- isotonic regression;
- all predictors;
- a sensitivity condition excluding `PageValues`;
- probability accuracy, calibration, discrimination, and F1 performance.

## Dataset

**Online Shoppers Purchasing Intention Dataset**

- UCI dataset ID: 468
- DOI: https://doi.org/10.24432/C5F88Q
- Source: https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset
- Licence: CC BY 4.0

The raw dataset is not stored in this repository. Use the download script:

```bash
python scripts/download_data.py
```

The script downloads the official UCI archive and verifies the CSV using SHA-256.

## Experimental design

- 12,330 original sessions
- 125 exact duplicates removed before partitioning
- 12,205 sessions used for analysis
- 10 fixed random seeds: `1` to `10`
- Stratified 60% training / 20% calibration / 20% test split
- Five-fold stratified cross-validation on the training partition
- Hyperparameter selection by mean log loss
- Separate calibration partition
- Fixed F1 threshold of 0.50
- No oversampling, SMOTE, synthetic observations, or class weighting

### Models

**Logistic Regression**

- L2 regularisation
- `C ∈ {0.01, 0.1, 1, 10}`

**XGBoost**

- `max_depth ∈ {3, 5}`
- `learning_rate ∈ {0.05, 0.10}`
- `n_estimators ∈ {200, 400}`
- `subsample ∈ {0.8, 1.0}`
- `colsample_bytree ∈ {0.8, 1.0}`

## Evaluation metrics

- Brier score
- Log loss
- Expected calibration error using 10 equal-width bins
- ROC-AUC
- PR-AUC using average precision
- Positive-class F1-score at 0.50

## Environment

The reference environment used:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- SciPy 1.17.0
- scikit-learn 1.8.0
- XGBoost 3.1.3
- Matplotlib 3.10.8

Install the exact dependencies with:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate purchase-intention-calibration
```

The experiment performs a startup check for the two versions most likely to affect model reproducibility: **scikit-learn 1.8.0** and **XGBoost 3.1.3**. If either version differs, the run stops with a clear error rather than producing results under a different environment.

For reproducibility across operating systems and reviewer machines, hyperparameter search is executed with a single worker (`GridSearchCV(n_jobs=1)`). This changes only task scheduling; the folds, parameter grids, scoring criterion and selected models remain unchanged.

## Run the experiment

Download the dataset:

```bash
python scripts/download_data.py
```

Run the full experiment:

```bash
python src/run_experiment.py data/online_shoppers_intention.csv --output-dir outputs
```

Verify a fresh run against the included reference outputs:

```bash
python scripts/verify_results.py --output-dir outputs
```

On Windows, the complete workflow can also be run with:

```text
run_all.bat
```

On macOS/Linux:

```bash
bash run_all.sh
```

## Repository structure

```text
.
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   ├── METHODS.md
│   └── OUTPUTS.md
├── scripts/
│   ├── download_data.py
│   └── verify_results.py
├── src/
│   └── run_experiment.py
├── outputs/
└── reference_results/
```

## Notes

The confidence intervals summarise variability across the ten repeated stratified holdout partitions. Because test partitions can overlap, they should not be interpreted as independent population-level inferential intervals.

The all-predictor experiment uses complete session-level records. The `PageValues` exclusion is a sensitivity analysis designed to assess dependence on that predictor.
