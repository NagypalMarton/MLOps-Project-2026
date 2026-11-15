# HW2: Data validation and evaluation gates

## R1: Two Pandera contracts

[`src/mlops_project/schemas.py`](../src/mlops_project/schemas.py) defines two
strict `DataFrameModel` contracts:

- `Ingestion` describes the raw 32-column PMS export. It keeps documented
  missing values, accepts export sentinels such as `Undefined`, and accepts the
  observed `adr=-6.38` anomaly. Every numeric column still has a defensible
  domain or corruption upper bound.
- `ModelInput` describes the cleaned 31-column frame used before feature
  encoding. It removes the two post-booking leakage columns, requires missing
  identifiers to be filled, requires `adr >= 0`, and requires
  `total_guests > 0`.

The stricter model-input rules are intentional: a negative price and a booking
with no guests are raw-export quirks that must not reach the classifier.

The observed dataset has 119,390 rows and 32 columns. It contains four missing
`children` values, 488 missing countries, 16,340 missing agents, 112,593
missing companies, one negative ADR, and 180 rows with zero adults, children,
and babies. Cleaning produces 119,210 model-input rows. Both contracts are
validated in the data preparation path, and the schema tests cover strict
columns, negative ADR rejection, and zero-guest rejection.

## R2: Validation gate in the DVC pipeline

The `validate` stage in [`dvc.yaml`](../dvc.yaml) runs the `Ingestion` contract
and writes the cache-disabled verdict
[`reports/validation.json`](../reports/validation.json). The `prepare` stage
declares that verdict as a dependency; this is the gate edge. A validation
failure exits with code 1, so DVC stops before `prepare`.

The failure demonstration is:

```text
> uv run python scripts/demo_validation_gate.py
Planted invalid lead_time=999999 in the first data row.
Running stage 'validate':
> python -m mlops_project.validate
Data validation failed; see reports/validation.json.
ERROR: failed to reproduce 'validate': failed to run:
python -m mlops_project.validate, exited with 1
dvc repro validate exited with code 255.
Restored data/hotel_bookings.csv exactly.
restored_md5=20E00E5BA40F5EE6BCD9B87FBF2670F9
```

The resulting report contained:

```json
{
  "valid": false,
  "rows": 119390,
  "columns": 32,
  "error": "Column 'lead_time' failed element-wise validator number 1: less_than_or_equal_to(1000) failure cases: 999999"
}
```

After restoration, `uv run dvc repro` completed the validation and preparation
stages, and the second run reported `Data and pipelines are up to date`.

## R3: Slice evaluation

The `evaluate` stage writes aggregate metrics and the cache-disabled
[`metrics/slices.json`](../metrics/slices.json). The two slice definitions live
in [`params.yaml`](../params.yaml):

- `hotel`: City Hotel versus Resort Hotel. Operations teams care because the
  properties have different booking mixes and cancellation patterns.
- `deposit_type`: No Deposit, Non Refund, and Refundable. Revenue and risk
  teams care because deposit policy changes the financial exposure of a
  cancellation.

Each row contains `n`, `positive_labels`, and the classification metrics:

| slice | value | n | positive labels | F1 |
|---|---|---:|---:|---:|
| hotel | City Hotel | 19,648 | 8,303 | 0.7755 |
| hotel | Resort Hotel | 10,155 | 2,747 | 0.8023 |
| deposit_type | No Deposit | 26,108 | 7,399 | 0.7445 |
| deposit_type | Non Refund | 3,661 | 3,642 | 0.9955 |
| deposit_type | Refundable | 34 | 9 | 0.2767 |

The slice columns are copied from the raw rows during `prepare`, before the
feature encoder fills missing values or applies scaling. If slicing happened
after preprocessing, the stakeholder-facing categories would be lost or
replaced by encoded numbers, and imputation could move a missing booking into a
group without making that change visible in the slice table.

## R4: Acceptance criteria

The `acceptance` block in [`params.yaml`](../params.yaml) is versioned as
`hw2-r4-v1`. The current measured test-set aggregate is accuracy `0.7927` and
F1 `0.7846`. The absolute floors are therefore:

- accuracy `>= 0.75`: a margin of `0.0427` below the observed model;
- F1 `>= 0.70`: a margin of `0.0846` below the observed model.

The floors are also above the trivial constant predictors. On the 29,803-row
test set, always predicting zero scored accuracy `0.6292` and weighted F1
`0.4860`; always predicting one scored accuracy `0.3708` and weighted F1
`0.2006`. Thus both absolute floors reject a constant classifier rather than
merely accepting the class-frequency baseline.

The recorded regression baseline is
[`metrics/baseline.json`](../metrics/baseline.json), refreshed from the
current `hotel-cancel-predictor@staging` champion: F1 `0.7846`. F1 is the
named primary metric. The tolerance is `0.01`, so the regression floor is
`0.7746`; the current `0.7846` matches the recorded champion and passes.

For slices, the F1 floor is `0.70` with `min_support=100`. The smallest
supported slice is `deposit_type=Non Refund` with `n=3,661` and F1 `0.9955`;
the smallest defined slice is `Refundable` with only `n=34`, so it is reported
as skipped rather than scored. The supported `No Deposit` slice has the lowest
F1 at `0.7445`, which motivates the `0.70` floor.

The criteria were selected after measuring this HW2 model and its slices, which
is an inversion of the ideal process: a production project should agree on
thresholds before seeing the candidate model. This limitation is recorded
explicitly rather than pretending that these thresholds were pre-registered.

## R5: Gate stage

The `gate` stage reads the aggregate metrics, slice report, recorded baseline,
and acceptance block, then writes the cache-disabled
[`reports/gate.json`](../reports/gate.json). Its pure decision function is
`evaluate_gate`; each check has `pass`, `fail`, or `skipped` status. Missing
metrics/baselines and slices below `min_support` are skipped with a reason.
The command prints the counts without opening the JSON:

```text
gate_status=pass passed=7 failed=0 skipped=1
```

The `Refundable` slice is the one skipped check. A failing floor or regression
check exits non-zero, so `dvc repro` stops. The verdict is deterministic: it
contains no timestamp, is written with sorted keys, and two unchanged gate
runs produce byte-identical JSON.

The NO-GO path was exercised by temporarily changing the recorded aggregate F1
from `0.7846` to `0.6000` and running the exact gate command from the DVC
stage:

```text
temporary_f1=0.6000
gate_status=fail passed=5 failed=2 skipped=1
unexpected_exit=1
gate_status=pass passed=7 failed=0 skipped=1
restored_gate_exit=0
```

The original metrics file and passing verdict were restored after the
demonstration.

## R6: The gate has teeth

Training and publication are separate paths. The train stage always leaves
`model/model.pkl` and logs the candidate to MLflow, while
[`src/mlops_project/publish.py`](../src/mlops_project/publish.py) reads
`reports/gate.json` before creating a registry version or assigning the
`staging` alias. The `gate` stage calls publication only after a `pass`; a
`fail` exits before publication. This preserves a failed model artifact for
diagnosis instead of deleting it.

The reproducible demonstration command is:

```text
uv run python scripts/demo_model_gate.py
```

It temporarily changes `model.max_iter` from `1000` to `1`, runs the complete
pipeline, and restores the parameter afterward. The observed output was:

```text
gate_status=fail passed=6 failed=1 skipped=1
dvc_repro_exit=255
Model publication refused: reports/gate.json is not a GO verdict (status='fail').
publish_exit=1
model_exists=True
gate_status=pass passed=7 failed=0 skipped=1
published model=hotel-cancel-predictor version=4 alias=staging run_id=364ef4d699dd460ca9551489105fe51b
restored_dvc_repro_exit=0
```

The failed run therefore stops at the gate, the independent publication
command refuses it, and the trained `model/model.pkl` remains available.
Restoration returns the committed state to GO.

## R7: The recorded baseline

The committed [`metrics/baseline.json`](../metrics/baseline.json) contains the
champion metrics consumed by the gate. It records the source alias and linked
MLflow run ID as well as the `accuracy` and primary `f1` metric:

```json
{
  "accuracy": 0.7927,
  "f1": 0.7846,
  "model_version": "24",
  "run_id": "002eb38c227248e88ea04ee3a012f789",
  "source": "hotel-cancel-predictor@staging"
}
```

Refresh it from the registry with the alias-based command below; no model
version number is used to locate the champion:

```text
uv run python -m mlops_project.refresh_baseline
refreshed baseline from hotel-cancel-predictor@staging run_id=002eb38c227248e88ea04ee3a012f789 f1=0.7846
```

The no-baseline path is also explicit. If `metrics/baseline.json` is absent,
the gate writes a verdict whose regression check has status `skipped`, reports
the skipped count, gives the overall verdict status `skipped`, and exits
non-zero. It therefore cannot silently become a GO or publish a model:

```text
gate_status=skipped passed=6 failed=0 skipped=2
```

## R8: Fairness and subgroup check

The evaluation reports a **selection-rate gap** across the `deposit_type`
groups in [`metrics/fairness.json`](../metrics/fairness.json). Selection rate
is the fraction of bookings for which the model predicts cancellation. This
axis is useful here because a high predicted-cancellation rate changes how
revenue and operations teams allocate review effort and cancellation
mitigation; it is more directly actionable for this classifier than choosing a
generic aggregate gap without a stakeholder interpretation.

The measured report is:

```json
{
  "slice": "deposit_type",
  "eligible_groups": ["No Deposit", "Non Refund"],
  "selection_rate_gap": 0.8264
}
```

The `Refundable` group has only `n=34` test rows, below the configured
`min_support=100`, so it is reported with its rate but excluded from the
stable gap calculation. The gap is **reported, not gated**. The project has
not established a policy target for equal selection rates, and deposit policy
is a business-risk axis rather than a protected attribute. Turning this
observational check into a GO/NO-GO rule without a stakeholder-approved
policy could reject useful models or hide the small-slice uncertainty.

One group I would want to audit but cannot is customer nationality, especially
potentially vulnerable or protected nationalities. The dataset contains a
`country` booking field, but it is not a validated person-level protected
attribute, has substantial missingness, and does not establish that the
booking guest belongs to that group. A legitimate audit would require a
documented, consented protected-attribute field with an approved use and
enough support, collected during dataset design before training and evaluation;
that did not happen for this historical export.

## R9: What the gate cannot catch

These gates establish structural validity and measured test-set performance,
but they are not a complete production assurance system. The ingestion
contract cannot detect a plausible but semantically wrong value, a duplicated
booking, or a label that was silently changed upstream. The model gate cannot
detect future temporal drift, calibration failure, data leakage that produces
plausible test metrics, or a change in the business cost of false positives
and false negatives. The selection-rate report is also not a causal fairness
audit and cannot establish protected-group treatment because the dataset lacks
validated protected attributes. Finally, the gate does not prove that MLflow,
Silo, or the registry remain available after the run. Those concerns require
later drift monitoring, data lineage checks, calibration/cost analysis,
approved fairness data, and service/infrastructure monitoring rather than
another static threshold in this homework.

## R10: Tests

The gate decision is implemented as the plain-data
[`evaluate_gate`](../src/mlops_project/gate.py) function. The focused
[`tests/test_gate.py`](../tests/test_gate.py) suite covers a passing verdict
with a small slice skipped, a failing absolute floor, a missing-baseline
skipped verdict, and publication refusal. It does not read the dataset or
contact MLflow for the gate-logic cases.

With the Docker Compose services stopped, the focused suite passed:

```text
docker compose stop
uv run pytest tests/test_gate.py -q
....                                                                     [100%]
4 passed in 3.77s
```
