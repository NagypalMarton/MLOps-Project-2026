from mlops_project.gate import evaluate_gate
from mlops_project.publish import publish_model


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


def test_publish_refuses_non_go_verdict(tmp_path):
    verdict_path = tmp_path / "gate.json"
    run_path = tmp_path / "latest_run.json"
    verdict_path.write_text('{"status": "fail"}', encoding="utf-8")
    run_path.write_text('{"run_id": "unused"}', encoding="utf-8")

    try:
        publish_model(verdict_path, run_path)
    except RuntimeError as error:
        assert "publication refused" in str(error)
    else:
        raise AssertionError("A failed verdict must refuse publication.")
