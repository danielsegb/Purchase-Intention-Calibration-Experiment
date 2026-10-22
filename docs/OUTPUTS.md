# Generated outputs

Running `src/run_experiment.py` writes the following files to the selected output directory:

- `data_integrity_checks.json`: row counts, duplicate checks, missing values, and class distribution.
- `software_versions.json`: package and Python versions.
- `test_metrics_by_seed.csv`: test metrics for every seed, model, feature condition, and calibration condition.
- `summary_mean_sd_ci.csv`: mean, sample standard deviation, and 95% t interval for each metric.
- `tuning_results.csv`: selected hyperparameters and training cross-validation log loss.
- `tuning_parameter_frequencies.csv`: frequency of selected hyperparameter configurations.
- `paired_brier_contrasts.csv`: paired Brier-score contrasts.
- `pooled_reliability_data.csv`: ten-bin reliability data for the all-predictor configuration.
- `test_predictions_by_seed.csv`: test probabilities and observed outcomes.
- `reliability_logistic_regression.png`: Logistic Regression reliability diagram.
- `reliability_xgboost.png`: XGBoost reliability diagram.

The `reference_results/` directory contains a reference execution of the experiment. Use `scripts/verify_results.py` to compare a fresh run with these outputs.
