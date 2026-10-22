# Provenance and relationship to the manuscript analysis

`archive/original_analysis_script.py` is the retained analysis script associated with the manuscript results. It contains the exact modelling, splitting, tuning, calibration, metric, and summary logic used for the reported experiment.

`src/run_experiment.py` is the GitHub-facing reproducibility version. It preserves the same scientific protocol and adds only reproducibility/output conveniences:

- persistence of individual test predictions;
- automatic pooled reliability-data construction;
- automatic reliability-figure generation;
- tuning-parameter frequency output;
- all paired Brier contrasts used in the manuscript;
- expanded software-version recording.

The retained numerical outputs from the manuscript analysis are stored in `reference_results/`. A fresh run can be compared against them with `scripts/verify_results.py`.

No new model, feature-selection rule, resampling procedure, calibration method, threshold optimisation, or evaluation metric has been introduced into the scientific comparison.
