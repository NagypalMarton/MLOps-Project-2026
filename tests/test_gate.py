from mlops_project.gate import evaluate_gate


CRITERIA = {
    "criteria_version": "test-v1",
    "absolute_floors": {"accuracy": 0.75, "f1": 0.70},
    "regression": {
        "primary_metric": "f1",
        "tolerance": 0.01,
    },
    "slices": {"metric": "f1", "floor": 0.70, "min_support": 100},
}


def test_gate_passes_and_skips_small_slice():
    verdict = evaluate_gate(
        {"accuracy": 0.80, "f1": 0.78},
        [
            {"slice": "hotel", "value": "City Hotel", "n": 200, "f1": 0.75},
            {"slice": "deposit_type", "value": "Refundable", "n": 20, "f1": 0.10},
        ],
        {"f1": 0.78},
        CRITERIA,
    )

    assert verdict["status"] == "pass"
    assert verdict["failed"] == 0
    assert verdict["skipped"] == 1


def test_gate_fails_below_absolute_floor():
    verdict = evaluate_gate(
        {"accuracy": 0.80, "f1": 0.60},
        [],
        {"f1": 0.78},
        CRITERIA,
    )

    assert verdict["status"] == "fail"
    assert verdict["failed"] == 2


def test_gate_skips_missing_baseline():
    verdict = evaluate_gate(
        {"accuracy": 0.80, "f1": 0.78},
        [],
        None,
        CRITERIA,
    )

    assert verdict["status"] == "pass"
    assert verdict["skipped"] == 1
