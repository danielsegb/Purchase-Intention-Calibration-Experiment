# Generated outputs

Running `src/run_experiment.py` writes the following files to the selected output directory:

- `data_integrity_checks.json`: raw-data and duplicate-removal checks.
- `software_versions.json`: software versions used in the execution.
- `test_metrics_by_seed.csv`: all test metrics for every seed, model, feature condition, and calibration condition.
- `summary_mean_sd_ci.csv`: mean, sample SD, and 95% t interval for every metric.
- `tuning_results.csv`: selected hyperparameters and training-CV log loss by seed.
- `tuning_parameter_frequencies.csv`: frequency of selected hyperparameter configurations.
- `paired_brier_contrasts.csv`: paired Brier-score contrasts used in the manuscript.
- `pooled_reliability_data.csv`: pooled ten-bin reliability points for the all-predictor configuration.
- `reliability_logistic_regression.png`: manuscript reliability figure for Logistic Regression.
- `reliability_xgboost.png`: manuscript reliability figure for XGBoost.

The `reference_results/` directory contains the retained outputs from the analysis used to prepare the manuscript. `scripts/verify_results.py` compares a fresh run against these reference files.
