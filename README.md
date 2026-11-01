# My MLOps Project

Your semester-long project for **Lifecycle of Artificial Intelligence Systems**.
You start from this minimal, working baseline and grow it into a full MLOps
project across HW1–HW5 — see [`HOMEWORK.md`](HOMEWORK.md).

This repo is **separate from the course materials**. The course repo (lectures,
labs, datasets, homework briefs) you read and do the labs in; *this* repo is your
own project, on your own dataset, and is what gets graded.

## What's here

```
src/mlops_project/   config.py · data.py · model.py · cli.py   (a simple baseline pipeline)
src/main.py          entry point
tests/               smoke tests that pass on synthetic data (no dataset needed)
data/                put your dataset here (see data/README.md)
docs/                DATA_DICTIONARY.md (fill it in) + your proposal
.env.example         configuration — copy to .env
compose.yaml         Docker Compose file for local services (Postgres, Silo, MLflow)
Makefile             Convenience commands for infrastructure and pipeline
pyproject.toml       Project dependencies and metadata
uv.lock              Locked dependencies (committed)
```

The baseline trains a scikit-learn `StandardScaler` + (logistic or linear)
regression on a CSV you point it at — classification or regression, set by
`PROBLEM_TYPE`. It's deliberately simple; the course is about the lifecycle, not
the model.

## Quick start

### 1. Environment Setup (without infrastructure)
```bash
cp .env.example .env          # set DATA_PATH, TARGET_COLUMN, PROBLEM_TYPE
uv sync --all-groups
uv run pytest -v              # green immediately (synthetic data)
# drop your dataset in data/ and:
uv run python src/main.py     # baseline metrics on your data
```

### 2. With Local Services (for HW1+)
```bash
# Start infrastructure (PostgreSQL, Silo, MLflow)
make up                       # or: docker compose up -d --wait

# Initialize DVC (first time only)
make dvc-init                 # or: dvc init && dvc remote add -d silo s3://$(BUCKET_NAME)/datasets

# Execute the full pipeline
make repro                    # or: dvc repro

# View pipeline DAG
make dag                      # or: dvc dag

# Stop infrastructure when done
make down                     # or: docker compose down
```

## Next

1. Add your dataset to `data/` and document it in `docs/DATA_DICTIONARY.md`.
2. Write your one-page proposal in `docs/proposal.md` (Week 2 milestone).
3. Follow [`HOMEWORK.md`](HOMEWORK.md) from HW1 onward.

## Prerequisites
- `uv` (https://docs.astral.sh/uv/)
- Docker Engine
- Python 3.12+ (uv installs it)
