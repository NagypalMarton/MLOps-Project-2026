# My MLOps Project

Your semester-long project for **Lifecycle of Artificial Intelligence Systems**.
This repository contains the HW1 implementation for a full MLOps project;
later homework stages are described in [`HOMEWORK.md`](HOMEWORK.md).

This repo is **separate from the course materials**. The course repo (lectures,
labs, datasets, homework briefs) you read and do the labs in; *this* repo is your
own project, on your own dataset, and is what gets graded.

## What's here

```
src/mlops_project/   pipeline, MLflow tracking, and the runs-for-data CLI
tests/               smoke tests that pass on synthetic data (no dataset needed)
data/                put your dataset here (see data/README.md)
docs/                DATA_DICTIONARY.md (fill it in) + your proposal
.env.example         configuration — copy to .env
compose.yaml         Docker Compose file for local services (Postgres, Silo, MLflow)
Makefile             Convenience commands for infrastructure and pipeline
pyproject.toml       Project dependencies and metadata
uv.lock              Locked dependencies (committed)
.dvc/config          DVC configuration (committed)
data/hotel_bookings.csv.dvc  DVC pointer file (committed)
```

The project predicts whether a hotel booking will be cancelled. It uses
scikit-learn `StandardScaler` and logistic regression, with categorical
encoding, missing-value handling, and leakage-column removal.

## Dataset Information (Tracked with DVC)

- **Pointer file**: `data/hotel_bookings.csv.dvc`
- **MD5 hash**: `20e00e5ba40f5ee6bcd9b87fbf2670f9`
- **Size**: 17,055,512 bytes (~17.1 MB)
- **Dimensions**: 119,390 rows × 32 columns
- **Target column**: `is_canceled` (binary classification)

## Quick start

### 1. Environment Setup (without infrastructure)
```bash
# PowerShell: Copy-Item .env.example .env
# POSIX shells: cp .env.example .env
uv sync --all-groups
uv run pytest -v              # green immediately (synthetic data)
```

### 2. With Local Services (for HW1+)
```bash
# Start infrastructure (PostgreSQL, Silo, MLflow)
make up                       # or: docker compose up -d --wait

# Execute the full pipeline
make repro                    # or: dvc repro

# Run only the ingestion validation gate
make validate                 # or: dvc repro validate

# Demonstrate rejection and exact restoration of invalid input
make validation-demo

# Find MLflow runs trained on the current DVC dataset version
runs-for-data

# View pipeline DAG
make dag                      # or: dvc dag

# Stop infrastructure when done
make down                     # or: docker compose down
```

`docker compose up -d --wait` builds the current MLflow image, starts Postgres
and Silo, and creates the `mlflow-artifacts` and `dvc-data` buckets idempotently.
MLflow is available at
`http://localhost:5000`; Silo's API and console are at ports `9000` and `9001`.
The Silo remote uses `http://localhost:9000` from the host.

## Next

1. Add your dataset to `data/` and document it in `docs/DATA_DICTIONARY.md`.
2. Write your one-page proposal in `docs/proposal.md` (Week 2 milestone).
3. Follow [`HOMEWORK.md`](HOMEWORK.md) from HW1 onward.

## Prerequisites
- `uv` (https://docs.astral.sh/uv/)
- Docker Engine
- Python 3.12+ (uv installs it)
