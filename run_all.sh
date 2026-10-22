#!/usr/bin/env bash
set -euo pipefail
python scripts/download_data.py
python src/run_experiment.py data/online_shoppers_intention.csv --output-dir outputs
python scripts/verify_results.py --output-dir outputs
