from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

MODEL_NAME = "hotel-cancel-predictor"
ALIAS = "staging"
BASELINE_PATH = Path("metrics/baseline.json")


def refresh_baseline(
    output_path: Path = BASELINE_PATH,
    model_name: str = MODEL_NAME,
    alias: str = ALIAS,
) -> dict[str, object]:
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri or not tracking_uri.strip():
        raise EnvironmentError("MLFLOW_TRACKING_URI must be configured.")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, alias)
    if not version.run_id:
        raise RuntimeError(
            f"MLflow alias {model_name}@{alias} has no linked run_id."
        )
    run = client.get_run(version.run_id)
    metrics = {
        key: run.data.metrics[key]
        for key in ("accuracy", "f1")
        if key in run.data.metrics
    }
    if "f1" not in metrics:
        raise RuntimeError(
            f"Run {version.run_id} has no f1 metric for the baseline."
        )

    baseline: dict[str, object] = {
        "source": f"{model_name}@{alias}",
        "run_id": version.run_id,
        "model_version": str(version.version),
        **metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"refreshed baseline from {model_name}@{alias} "
        f"run_id={version.run_id} f1={metrics['f1']:.4f}"
    )
    return baseline


def main() -> None:
    refresh_baseline()


if __name__ == "__main__":
    main()
