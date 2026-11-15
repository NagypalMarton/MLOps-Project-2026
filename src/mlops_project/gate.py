from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

VERDICT_PATH = Path("reports/gate.json")


def _check(
    name: str,
    actual: float | None,
    threshold: float | None,
    passed: bool | None,
    reason: str | None = None,
) -> dict[str, Any]:
    result = {
        "name": name,
        "status": "skipped" if passed is None else "pass" if passed else "fail",
        "actual": actual,
        "threshold": threshold,
    }
    if reason:
        result["reason"] = reason
    return result


def evaluate_gate(
    metrics: dict[str, float],
    slices: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for metric, floor in acceptance["absolute_floors"].items():
        actual = metrics.get(metric)
        checks.append(
            _check(
                f"absolute_floor.{metric}",
                actual,
                floor,
                None if actual is None else actual >= floor,
                None if actual is not None else "metric is missing",
            )
        )

    regression = acceptance["regression"]
    primary_metric = regression["primary_metric"]
    current = metrics.get(primary_metric)
    baseline_value = baseline.get(primary_metric) if baseline else None
    tolerance = regression["tolerance"]
    checks.append(
        _check(
            f"regression.{primary_metric}",
            current,
            None if baseline_value is None else baseline_value - tolerance,
            None
            if current is None or baseline_value is None
            else current >= baseline_value - tolerance,
            (
                "baseline or current metric is missing"
                if current is None or baseline_value is None
                else None
            ),
        )
    )

    slice_rule = acceptance["slices"]
    for row in slices:
        name = f"slice.{row['slice']}={row['value']}.{slice_rule['metric']}"
        if row["n"] < slice_rule["min_support"]:
            checks.append(
                _check(
                    name,
                    row.get(slice_rule["metric"]),
                    slice_rule["floor"],
                    None,
                    f"support {row['n']} is below min_support "
                    f"{slice_rule['min_support']}",
                )
            )
        else:
            actual = row.get(slice_rule["metric"])
            checks.append(
                _check(
                    name,
                    actual,
                    slice_rule["floor"],
                    None if actual is None else actual >= slice_rule["floor"],
                    None if actual is not None else "slice metric is missing",
                )
            )

    failed = sum(check["status"] == "fail" for check in checks)
    skipped = sum(check["status"] == "skipped" for check in checks)
    passed = sum(check["status"] == "pass" for check in checks)
    baseline_missing = baseline is None
    status = (
        "fail"
        if failed
        else "skipped"
        if baseline_missing or not passed
        else "pass"
    )
    return {
        "criteria_version": acceptance["criteria_version"],
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "checks": checks,
    }


def main() -> None:
    metrics = json.loads(Path("metrics/metrics.json").read_text(encoding="utf-8"))
    slices = json.loads(Path("metrics/slices.json").read_text(encoding="utf-8"))
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    acceptance = dict(params["acceptance"])
    acceptance["criteria_version"] = params["criteria_version"]
    baseline_path = Path(acceptance["regression"]["baseline_file"])
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.exists()
        else None
    )
    verdict = evaluate_gate(metrics, slices, baseline, acceptance)
    VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"gate_status={verdict['status']} "
        f"passed={verdict['passed']} "
        f"failed={verdict['failed']} "
        f"skipped={verdict['skipped']}"
    )
    if verdict["status"] in {"fail", "skipped"}:
        raise SystemExit(1)
    if verdict["status"] == "pass":
        from .publish import publish_model

        publish_model()


if __name__ == "__main__":
    main()
