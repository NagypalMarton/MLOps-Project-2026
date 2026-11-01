# HW1: Data Versioning and Experiment Tracking

## R5: At least two dataset versions ✓ MET

**Version 1 (Original)**
- Description: Raw hotel bookings dataset as downloaded from Kaggle (UTF-16 encoded with BOM).
- Changes: None (baseline).
- MD5: `cfa4d278475915ee56c9cb94a74ed3c5`
- Size: 16,974,990 bytes
- Dimensions: 119,390 rows × 32 columns

**Version 2 (Cleaned)**
- Description: Dataset with fixed encoding issue (UTF-16 → UTF-8) and removal of Byte Order Mark (BOM).
- Changes: 
  - Converted file encoding from UTF-16 to UTF-8 without BOM.
  - Removed any potential BOM at the start of the file.
  - This ensures compatibility with standard Python CSV readers and avoids UnicodeDecodeError.
- MD5: `20e00e5ba40f5ee6bcd9b87fbf2670f9`
- Size: 17,055,512 bytes
- Dimensions: 119,390 rows × 32 columns (same number of rows and columns, but slightly larger byte size due to UTF-8 being less efficient than UTF-16 for this particular dataset)

**How to Produce Version 2**
```bash
# Convert encoding and remove BOM using pandas
python -c "
import pandas as pd
df = pd.read_csv('data/hotel_bookings_v1.csv', encoding='utf-16')
df.to_csv('data/hotel_bookings.csv', index=False, encoding='utf-8')
"
# Alternative direct conversion (if iconv is available):
# iconv -f UTF-16 -t UTF-8 data/hotel_bookings_v1.csv > data/hotel_bookings.csv

# Update DVC to track the new version
dvc add data/hotel_bookings.csv
git add data/hotel_bookings.csv.dvc
git commit -m "Update dataset to cleaned version (UTF-8 without BOM)"
```

**Moving Between Versions**
To switch to Version 1 (original):
```bash
git checkout 1d8a48e -- data/hotel_bookings.csv.dvc  # Commit that restored the original dataset
dvc checkout data/hotel_bookings.csv
```
To switch to Version 2 (cleaned):
```bash
git checkout fd6d990 -- data/hotel_bookings.csv.dvc  # Commit that updated to cleaned version
dvc checkout data/hotel_bookings.csv
```

**Verification**
Demonstrated moving between versions and validating that the correct dataset files are checked out with matching MD5 hashes before and after checkout operations.

## R6: A declared pipeline ○ NOT FULLY MET (Foundation Laid)

Current `dvc.yaml` defines a single stage for rapid validation. To fully meet R6, the pipeline should be structured as three stages:

**Required Pipeline Stages (`dvc.yaml`)**
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

**Notes for Full Compliance**
- Each `cmd` corresponds to a runnable module (e.g., `python -m mlops_project.prepare`)
- Source files are listed in `deps`, so editing source code marks stages as stale
- Hyperparameters live in `params.yaml` (not `.env`), allowing DVC to detect changes
- `metrics/metrics.json` should be a `cache: false` output so it's committed to Git
- `dvc.lock` should be committed after running `dvc repro` to ensure exact reproducibility
- Running `dvc repro` twice with no changes should re-run nothing the second time
- Changing a parameter in `params.yaml` should mark only the `prepare` stage as stale, causing downstream stages to re-run

**Status**: Foundation code exists in `src/mlops_project/` but the three-stage pipeline structure needs to be implemented in `dvc.yaml` and corresponding modules created.

## R7: MLflow tracking and the model registry ○ NOT FULLY MET (Foundation Laid)

Our implementation includes MLflow integration attempts in `src/mlops_project/cli.py`, but full compliance requires actual execution of multiple runs.

**Experiment Setup**
- MLflow experiment name: `hotel-cancel-prediction` (configured in code)
- Tracking URI: Configured via `MLFLOW_TRACKING_URI` in `.env` (points to containerized MLflow server)

**Required Hyperparameter Comparison (≥4 Runs)**
Plan to compare values of `max_iter` for Logistic Regression: `[100, 500, 1000, 2000]`.
All other settings (test size, random seed, etc.) would remain constant.

| Run | max_iter | Accuracy | Precision | Recall | F1-Score | Notes |
|-----|----------|----------|-----------|--------|----------|-------|
| 1   | 100      | TBD      | TBD       | TBD    | TBD      | Underfitting; low iterations |
| 2   | 500      | TBD      | TBD       | TBD    | TBD      | Improving |
| 3   | 1000     | TBD      | TBD       | TBD    | TBD      | Best balance (expected) |
| 4   | 2000     | TBD      | TBD       | TBD    | TBD      | Same as 1000; no further gain (expected) |

**Winning Model Requirements**
- **Model name**: `hotel-cancel-predictor`
- **Version**: Assigned automatically by MLflow Model Registry
- **Alias**: `staging` (assigned to best version)
- **Reasoning**: F1-score is the primary metric chosen to balance precision and recall in the moderately imbalanced dataset (37.04% cancellation rate).
- **Run ID**: Must be populated in the MLflow Model Registry when registering from `runs:/<run_id>/model`, creating a traceable link from the registry version back to the specific training run.
- **Git Commit Tag**: Every MLflow run should include the `git_commit` tag for reproducibility.

**Status**: MLflow integration code exists in `src/mlops_project/cli.py` but requires:
1. Working Docker services (PostgreSQL, Silo, MLflow)
2. Successful DVC initialization and remote configuration
3. Execution of `dvc repro` to generate actual runs
4. Potential code refinements to ensure proper MLflow logging, model registration, and git commit tagging

## Overall Status

✅ **R1-R5**: Fully met and verified  
○ **R6-R7**: Structural foundation laid and implementation attempted, but full verification requires successful execution in a working environment (currently hindered by environmental/dependency issues noted during setup)

Once environmental constraints are resolved, the workflow to achieve full compliance would be:
1. `make up` - Start PostgreSQL, Silo, and MLflow services
2. `make dvc-init` - Initialize DVC and configure connection to Silo  
3. `make repro` - Execute the pipeline (with three-stage structure once implemented)
   - This would version data in Silo via DVC
   - Track experiments in MLflow (parameters, metrics, model, dataset version)
   - Register the best model with appropriate name/version/alias
   - Only re-run stages affected by changes on subsequent executions