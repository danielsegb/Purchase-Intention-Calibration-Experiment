# Data

This repository does **not** redistribute the raw dataset by default.

The experiment uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository:

- Creators: C. Sakar and Yomi Kastro
- UCI dataset ID: 468
- DOI: https://doi.org/10.24432/C5F88Q
- Licence: CC BY 4.0
- Original rows: 12,330
- Predictors: 17
- Target: `Revenue`

Run:

```bash
python scripts/download_data.py
```

The script downloads the official UCI archive and extracts:

`data/online_shoppers_intention.csv`

The exact raw CSV used for the reported analysis has SHA-256:

`b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`

The analysis script independently checks row count, missing values, duplicates, and the post-deduplication class distribution before model fitting.
