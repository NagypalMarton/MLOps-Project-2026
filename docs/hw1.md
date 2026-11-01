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
> git clone <repository-url> hw1-fresh-clone
> cd hw1-fresh-clone
> uv sync --all-groups
> uv run dvc pull
1 file added and 2 files fetched
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

## R7: MLflow tracking and registry

The named experiment is `hotel-cancel-prediction`. Each run logs the settings,
`params.yaml`, dataset pointer hash, metrics, model, plots, and a `git_commit`
tag. Four comparison runs were executed:

| max_iter | F1 | run_id |
|---:|---:|---|
| 100 | 0.786 | `be31a762fa6c4c5098f189482f90caec` |
| 500 | 0.786 | `8620fd7a792d47d2b324cfb9883f1660` |
| 1000 | 0.786 | `e2d98b662ebf4f2b9a8341821b63b49f` |
| 2000 | 0.786 | `167ebb6f1f2e42fe8aac915adb3bf120` |

F1 is the primary metric because cancellation classes are imbalanced and the
metric balances precision and recall. The registered model is
`hotel-cancel-predictor`; version `6` is assigned the `staging` alias and its
`run_id` is `167ebb6f1f2e42fe8aac915adb3bf120`.

The DVC `train` stage uses the same MLflow logging path as the CLI, so pipeline
training runs also receive `dvc_md5`, `git_commit`, parameters, metrics, and a
logged model. The Lab 4 lookup command is:

```text
> runs-for-data
Searching for runs with tags.dvc_md5 = '20e00e5ba40f5ee6bcd9b87fbf2670f9'
run_id tags.mlflow.runName params.max_iter metrics.f1 tags.git_commit
...

8 run(s).
```
