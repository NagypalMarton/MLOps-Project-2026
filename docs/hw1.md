# HW1: Data Versioning and Experiment Tracking

## R3-R4: DVC and Silo

The committed pointer `data/hotel_bookings.csv.dvc` names a 119,390-row,
32-column dataset with:

```text
md5: 20e00e5ba40f5ee6bcd9b87fbf2670f9
size: 17055512
```

`.dvc/config` contains only the default Silo remote and its endpoint; credentials
are supplied through the ignored `.env` file. After the stack is running:

```text
> $env:AWS_ACCESS_KEY_ID="minioadmin"
> $env:AWS_SECRET_ACCESS_KEY="minioadmin"
> $env:AWS_ENDPOINT_URL="http://localhost:9000"
> uv run dvc push
3 files pushed
```

The DVC bucket is separate from the MLflow artifact bucket. The object is stored
in Silo under the DVC layout
`files/md5/20/e00e5ba40f5ee6bcd9b87fbf2670f9`.

Fresh-clone verification:

```text
> git clone https://github.com/NagypalMarton/MLOps-Project-2026.git hw1-fresh-clone
> cd hw1-fresh-clone
> uv sync --all-groups
> uv run dvc pull
3 files fetched and 3 files added
> Get-FileHash data/hotel_bookings.csv -Algorithm MD5
Hash: 20E00E5BA40F5EE6BCD9B87FBF2670F9
```

## R5: Dataset versions

Version 1 is the original UTF-16/BOM file; version 2 is the UTF-8 file tracked
by the current pointer. The content hashes and sizes are intentionally
different. To demonstrate switching, restore the pointer from the desired
commit, then run:

```text
git checkout 1d8a48e -- data/hotel_bookings.csv.dvc
uv run dvc checkout data/hotel_bookings.csv
md5 before: 20e00e5ba40f5ee6bcd9b87fbf2670f9
md5 after:  cfa4d278475915ee56c9cb94a74ed3c5
```

To switch back to Version 2:

```text
git checkout fd6d990 -- data/hotel_bookings.csv.dvc
uv run dvc checkout data/hotel_bookings.csv
md5 before: cfa4d278475915ee56c9cb94a74ed3c5
md5 after:  20e00e5ba40f5ee6bcd9b87fbf2670f9
```

## R6: Declared DVC pipeline

`dvc.yaml` contains executable `prepare`, `train`, and `evaluate` stages.
`params.yaml` is a dependency of `train`; `metrics/metrics.json` is a
non-cached metrics output and `dvc.lock` records the exact stage state.

After changing the pipeline source once, the first reproduction rebuilt the
affected stages. The following reproduction was a no-op:

```text
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
```

Changing `model.max_iter` in `params.yaml` marks `train` stale and causes
`evaluate` to follow it; `prepare` remains cached.

For example, changing `max_iter` from 1000 to 500 produced:

```text
Stage 'prepare' didn't change, skipping
Running stage 'train':
> python -m mlops_project.train
Updating lock file 'dvc.lock'
Running stage 'evaluate':
> python -m mlops_project.evaluate
```

## R7: MLflow tracking and registry

The named experiment is `hotel-cancel-prediction`. Each run logs the settings,
`params.yaml`, dataset pointer hash, metrics, model, plots, and a `git_commit`
tag. Four comparison runs were executed:

| max_iter | F1 | run_id |
|---:|---:|---|
| 100 | 0.786 | `c36b952ce5294bc4a80cf994cfe6e007` |
| 500 | 0.786 | `ae7c1f9c5eb347f6a16abd965700b2d1` |
| 1000 | 0.786 | `b6b064cf86ce495f81a5757da77801d5` |
| 2000 | 0.786 | `b307a2d0ad39470997d8f38d7abacdf7` |

F1 is the primary metric because cancellation classes are imbalanced and the
metric balances precision and recall. The registered model is
`hotel-cancel-predictor`; version `12` is assigned the `staging` alias and its
`run_id` is `b307a2d0ad39470997d8f38d7abacdf7`.

The DVC `train` stage uses the same MLflow logging path as the CLI, so pipeline
training runs also receive `dvc_md5`, `git_commit`, parameters, metrics, and a
logged model. Each run also receives `dvc_remote_uri=s3://dvc-data`, which is
the default remote in `.dvc/config` (with endpoint
`http://localhost:9000`). The Lab 4 lookup command is:

```text
> runs-for-data
Searching for runs with tags.dvc_md5 = '20e00e5ba40f5ee6bcd9b87fbf2670f9'
run_id tags.mlflow.runName params.max_iter metrics.f1 tags.git_commit tags.dvc_remote_uri
adc69a21eb1649e8916a6d48707c2639 exultant-finch-248 1000 0.786 6c508a728d91923241d506216e351a2849c4c646 s3://dvc-data
c36b952ce5294bc4a80cf994cfe6e007 burly-gnat-966 100 0.786 02c6f8f7344a63d1db222cf018da2305b5b0fba4 s3://dvc-data
ae7c1f9c5eb347f6a16abd965700b2d1 resilient-fish-716 500 0.786 02c6f8f7344a63d1db222cf018da2305b5b0fba4 s3://dvc-data
b6b064cf86ce495f81a5757da77801d5 wistful-shrimp-121 1000 0.786 02c6f8f7344a63d1db222cf018da2305b5b0fba4 s3://dvc-data
b307a2d0ad39470997d8f38d7abacdf7 intrigued-wren-729 2000 0.786 02c6f8f7344a63d1db222cf018da2305b5b0fba4 s3://dvc-data

11 run(s).
```

## R8-R9: Traceability chain

The registered model can be followed end to end using the MLflow server and
the committed DVC files:

```text
hotel-cancel-predictor@staging
  -> model version 17
  -> run adc69a21eb1649e8916a6d48707c2639
  -> params.max_iter=1000, metrics.f1=0.786
  -> git commit 6c508a728d91923241d506216e351a2849c4c646
  -> data md5 20e00e5ba40f5ee6bcd9b87fbf2670f9
  -> DVC remote s3://dvc-data, endpoint http://localhost:9000
```

The alias resolves to version 17 in the Model Registry; that version exposes
the run ID. The run's parameters, metric, and tags are visible through its
MLflow run page or API. The commit contains the pointer whose MD5 is shown in
the run's `dvc_md5` tag, and `.dvc/config` maps the DVC remote URI to Silo.

## R10: Reflection

The hardest issue was making the current MLflow image work with both PostgreSQL
metadata and Silo artifacts. The initial container lacked the required
PostgreSQL/S3 drivers and an older cached image produced database schema errors.
I diagnosed this from the MLflow container logs and failed pipeline runs, then
fixed it by building the current MLflow image with `psycopg[binary]` and
`boto3`, and verified the complete stack with `docker compose up -d --wait`,
MLflow registration, and the `runs-for-data` lookup.
