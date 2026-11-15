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
