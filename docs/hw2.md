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
