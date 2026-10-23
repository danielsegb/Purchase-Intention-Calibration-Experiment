# Dataset

This directory contains the dataset used for the experiment: the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

- **UCI Dataset ID**: 468
- **DOI**: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)
- **Licence**: CC BY 4.0
- **Dimensions**: 12,330 sessions, 17 predictors, target: `Revenue`

### Download

The dataset can be downloaded and verified by running:

```bash
python scripts/download_data.py
```

This extracts `data/online_shoppers_intention.csv` and verifies its SHA-256 checksum:
`b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.
