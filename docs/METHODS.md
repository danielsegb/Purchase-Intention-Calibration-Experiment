# Experimental protocol

This document describes the computational protocol used in the experiment.

## Data preparation

1. Load the UCI Online Shoppers Purchasing Intention CSV.
2. Audit dimensions, missing values, target distribution, and exact duplicate rows.
3. Remove 125 exact duplicates **before** partitioning.
4. Retain 12,205 sessions: 10,297 non-purchase and 1,908 purchase sessions.
5. Convert `Revenue` to a binary outcome, with purchase = 1 and non-purchase = 0.
6. Treat `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, and `Weekend` as categorical predictors and one-hot encode them.
7. Standardise numerical predictors for Logistic Regression using training-only parameters. XGBoost receives numerical predictors without scaling.

## Feature conditions

The complete experiment is performed twice:

- **All predictors**: all 17 predictors.
- **Without PageValues**: the same data with only `PageValues` excluded.

No additional feature selection, SMOTE, oversampling, or class weighting is used.

## Repeated train-calibration-test partitions

Ten fixed random seeds are used: `1, 2, ..., 10`.

For each seed:

1. Stratified 80/20 split into development remainder and test data.
2. Stratified 75/25 split of the development remainder into training and calibration data.
3. This yields an effective **60% training / 20% calibration / 20% test** allocation.
4. Identical partitions are used for both models, both feature conditions, and all calibration methods.

## Logistic Regression

- L2 regularisation.
- Solver: `lbfgs`.
- `max_iter=2000`.
- Candidate `C`: `{0.01, 0.1, 1.0, 10.0}`.
- Five-fold stratified cross-validation on the training partition only.
- Selection criterion: mean log loss.
- No class weighting.

## XGBoost

- Objective: `binary:logistic`.
- Evaluation metric during fitting: log loss.
- Tree method: `hist`.
- `n_jobs=1` for deterministic CPU execution.
- Five-fold stratified cross-validation on the training partition only.
- Selection criterion: mean log loss.

Grid:

- `max_depth`: `{3, 5}`
- `learning_rate`: `{0.05, 0.10}`
- `n_estimators`: `{200, 400}`
- `subsample`: `{0.8, 1.0}`
- `colsample_bytree`: `{0.8, 1.0}`

## Probability calibration

Each tuned base estimator is evaluated under three probability conditions:

- Uncalibrated
- Platt scaling (`method="sigmoid"`)
- Isotonic regression (`method="isotonic"`)

The base estimator is frozen after training. Platt and isotonic calibrators are fitted **only** on the dedicated calibration partition. Test outcomes are not used for preprocessing, model selection, fitting, calibration, or threshold selection.

## Evaluation metrics

For every test partition:

- Brier score
- Log loss
- Expected calibration error (ECE), 10 equal-width probability bins
- ROC-AUC
- PR-AUC operationalised as average precision
- Positive-class F1-score at a fixed threshold of 0.50

## Statistical summaries

Across the ten repeated partitions, each metric is summarised using:

- arithmetic mean
- sample standard deviation
- 95% Student t interval with 9 degrees of freedom

Paired comparisons are calculated by random seed using unrounded repetition-level values. Because repeated test partitions overlap, these confidence intervals describe **split-to-split variability** and are not treated as independent population-level inference.

## Reliability diagrams

For the all-predictor configuration, test predictions from all ten repeated partitions are pooled separately by model and calibration condition. Ten equal-width probability bins are then used to plot mean predicted probability against observed purchase frequency. The dashed 45-degree line represents perfect calibration.
