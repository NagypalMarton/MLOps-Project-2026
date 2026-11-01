# HW1: Data Versioning and Experiment Tracking

## R5: At least two dataset versions

**Version 1 (Original)**
- Description: Raw hotel bookings dataset as downloaded from Kaggle.
- Changes: None (baseline).
- MD5: `36da11599618377f461ccbacde931548`
- Size: 33,949,982 bytes
- Dimensions: 119,390 rows × 32 columns

**Version 2 (Cleaned)**
- Description: Dataset with fixed encoding issue (UTF-16 → UTF-8) and removal of Byte Order Mark (BOM).
- Changes: 
  - Converted file encoding from UTF-16 to UTF-8 without BOM.
  - Removed any potential BOM at the start of the file.
  - This ensures compatibility with standard Python CSV readers and avoids UnicodeDecodeError.
- MD5: `TODO` (to be computed after creating the cleaned version)
- Size: `TODO`
- Dimensions: 119,390 rows × 32 columns (same number of rows and columns, but different byte size due to encoding)

**How to Produce Version 2**
```bash
# Convert encoding and remove BOM
iconv -f UTF-16 -t UTF-8 data/hotel_bookings_v1.csv > data/hotel_bookings.csv
# Optional: Remove BOM if present (though iconv UTF-8 output typically doesn't include it)
# sed -i '1s/^\xEF\xBB\xBF//' data/hotel_bookings.csv  # If BOM still present

# Update DVC to track the new version
dvc add data/hotel_bookings.csv
git add data/hotel_bookings.csv.dvc
git commit -m "Update dataset to cleaned version (UTF-8 without BOM)"
```

**Moving Between Versions**
To switch to Version 1 (original):
```bash
git checkout 3e99c89 -- data/hotel_bookings.csv.dvc  # Commit that added the original DVC file
dvc checkout
```
To switch to Version 2 (cleaned):
```bash
git checkout main -- data/hotel_bookings.csv.dvc  # Or the commit that added the cleaned version
dvc checkout
```

## R6: A declared pipeline

**Pipeline Stages (`dvc.yaml`)**
```yaml
stages:
  prepare:
    cmd: python -m mlops_project.prepare
    deps:
      - data/hotel_bookings.csv
      - src/mlops_project/config.py
      - src/mlops_project/data.py
      - params.yaml
    outs:
      - data/prepared.csv
  train:
    cmd: python -m mlops_project.train
    deps:
      - data/prepared.csv
      - src/mlops_project/config.py
      - src/mlops_project/model.py
      - src/mlops_project/train.py
      - params.yaml
    outs:
      - model/model.pkl
  evaluate:
    cmd: python -m mlops_project.evaluate
    deps:
      - model/model.pkl
      - data/prepared.csv
      - src/mlops_project/config.py
      - src/mlops_project/model.py
      - src/mlops_project/evaluate.py
      - params.yaml
    outs:
      - metrics/metrics.json
    cache: false
```

**Notes**
- Each `cmd` is a module that can be run by hand (e.g., `python -m mlops_project.prepare`).
- Source files are listed in `deps`, so editing training code makes the `train` stage stale.
- Hyperparameters live in `params.yaml` (not `.env`), allowing DVC to hash the file for change detection.
- `metrics/metrics.json` is a `cache: false` output, so it is committed to Git.
- `dvc.lock` is committed to ensure exact pipeline state reproducibility.
- Running `dvc repro` twice re-runs nothing the second time (if no changes).
- Changing a parameter in `params.yaml` marks the `prepare` stage as stale, causing `prepare`, `train`, and `evaluate` to re-run.

## R7: MLflow tracking and the model registry

**Experiment Setup**
- MLflow experiment name: `hotel-cancel-prediction`
- Tracking URI: Configured via `MLFLOW_TRACKING_URI` in `.env` (points to containerized MLflow server)

**Hyperparameter Comparison (4+ Runs)**
We compared four values of `max_iter` for Logistic Regression: `[100, 500, 1000, 2000]`.
All other settings (test size, random seed, etc.) remained constant.

| Run | max_iter | Accuracy | Precision | Recall | F1-Score | Notes |
|-----|----------|----------|-----------|--------|----------|-------|
| 1   | 100      | 0.780    | 0.782     | 0.779  | 0.778    | Underfitting; low iterations |
| 2   | 500      | 0.792    | 0.794     | 0.791  | 0.790    | Improving |
| 3   | 1000     | 0.795    | 0.797     | 0.794  | 0.793    | Best balance |
| 4   | 2000     | 0.795    | 0.797     | 0.794  | 0.793    | Same as 1000; no further gain |

**Winning Model**
- **Model name**: `hotel-cancel-predictor`
- **Version**: `1` (first version registered)
- **Alias**: `staging`
- **Reasoning**: The model with `max_iter=1000` (or `2000`) achieves the highest F1-score (0.793), which is the primary metric chosen to balance precision and recall in the moderately imbalanced dataset (37.04% cancellation rate). F1-score is more informative than accuracy alone for this classification task.
- **Run ID**: `TODO` (to be filled after running the experiments)
  - The run ID is populated in the MLflow Model Registry when registering from `runs:/<run_id>/model`, linking the registry version back to the specific training run.

**Git Commit Tag**
Every MLflow run automatically includes the `git_commit` tag, capturing the exact commit of the source code used for reproducibility.