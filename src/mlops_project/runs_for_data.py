from __future__ import annotations

import os
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def _dvc_md5(pointer: Path) -> str:
    with pointer.open(encoding="utf-8") as file:
        metadata = yaml.safe_load(file)
    return metadata["outs"][0]["md5"]


def main() -> None:
    pointer = Path("data/hotel_bookings.csv.dvc")
    dvc_md5 = _dvc_md5(pointer)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("hotel-cancel-prediction")
    if experiment is None:
        print(f"No experiment found for dvc_md5 = '{dvc_md5}'.")
        return

    print(f"Searching for runs with tags.dvc_md5 = '{dvc_md5}'")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.dvc_md5 = '{dvc_md5}'",
        order_by=["start_time ASC"],
    )
    print("run_id tags.mlflow.runName params.max_iter metrics.f1 tags.git_commit")
    for run in runs:
        print(
            run.info.run_id,
            run.data.tags.get("mlflow.runName", ""),
            run.data.params.get("max_iter", ""),
            run.data.metrics.get("f1", ""),
            run.data.tags.get("git_commit", ""),
        )
    print(f"\n{len(runs)} run(s).")


if __name__ == "__main__":
    main()
