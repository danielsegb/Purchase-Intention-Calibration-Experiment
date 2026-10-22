@echo off
python scripts\download_data.py || exit /b 1
python src\run_experiment.py data\online_shoppers_intention.csv --output-dir outputs || exit /b 1
python scripts\verify_results.py --output-dir outputs || exit /b 1
