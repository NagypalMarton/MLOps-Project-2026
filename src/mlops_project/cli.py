from __future__ import annotations

import json
import os
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path
import mlflow
import mlflow.sklearn as mlflow_sklearn

import yaml
from dataclasses import replace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .config import load_settings, Settings
from .data import build_dataset
from .model import build_model, evaluate_model


def _log_settings_as_params(settings: Settings) -> None:
    """Log settings as MLflow parameters, excluding the data_path object."""
    for key, value in vars(settings).items():
        if key == "data_path":
            # Log the data path as a string, but not the full Path object
            mlflow.log_param(key, str(value))
        else:
            mlflow.log_param(key, value)


def _log_params_yaml_as_params(params_path: str = "params.yaml") -> None:
    """Log parameters from params.yaml as MLflow parameters."""
    if not os.path.exists(params_path):
        return
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    def _flatten(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    flat_params = _flatten(params)
    for key, value in flat_params.items():
        mlflow.log_param(key, value)


def _get_dataset_version() -> str:
    """Extract dataset version hash from the DVC file."""
    dvc_file = "data/hotel_bookings.csv.dvc"
    if not os.path.exists(dvc_file):
        raise FileNotFoundError(f"DVC pointer not found: {dvc_file}")
    with open(dvc_file, "r") as f:
        meta = yaml.safe_load(f)
    return meta["outs"][0]["md5"]


def _get_dvc_remote_uri() -> str:
    config = ConfigParser()
    config.read(".dvc/config", encoding="utf-8")
    section = '\'remote "silo"\''
    if config.has_option(section, "url"):
        return config.get(section, "url")
    raise ValueError(f"DVC remote URL not found in {section}.")


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("Unable to determine the current Git commit.")


def _log_classification_plots(y_true, y_pred, y_pred_proba) -> None:
    """Log ROC curve and confusion matrix as MLflow artifacts and save to disk for DVC."""
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # ROC Curve
    roc_filename = "plots/roc_curve.png"
    RocCurveDisplay.from_predictions(y_true, y_pred_proba)
    plt.savefig(roc_filename, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(roc_filename)

    # Confusion Matrix
    cm_filename = "plots/confusion_matrix.png"
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(cm_filename, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_filename)


def _settings_from_params(settings: Settings) -> Settings:
    if not os.path.exists("params.yaml"):
        return settings
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}
    max_iter = params.get("model", {}).get("max_iter")
    if max_iter is None:
        return settings
    return replace(settings, max_iter=int(max_iter))


def log_mlflow_run(settings, model, metrics, y_test, y_pred, y_pred_proba) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri or not tracking_uri.strip():
        raise EnvironmentError("MLFLOW_TRACKING_URI must be configured.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("hotel-cancel-prediction")
    with mlflow.start_run() as run:
        _log_settings_as_params(settings)
        _log_params_yaml_as_params("params.yaml")
        dvc_md5 = _get_dataset_version()
        mlflow.log_param("dataset_version", dvc_md5)
        mlflow.set_tag("dvc_md5", dvc_md5)
        mlflow.set_tag("dvc_remote_uri", _get_dvc_remote_uri())
        mlflow.set_tag("git_commit", _get_git_commit())
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow_sklearn.log_model(model, "model")
        _log_classification_plots(y_test, y_pred, y_pred_proba)

        Path("reports").mkdir(parents=True, exist_ok=True)
        Path("reports/latest_run.json").write_text(
            json.dumps({"run_id": run.info.run_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        return run.info.run_id


def main() -> None:
    settings = _settings_from_params(load_settings())

    # Load and split data
    x_train, x_test, y_train, y_test = build_dataset(settings)

    # Build and train model
    model = build_model(settings)
    model.fit(x_train, y_train)

    # Evaluate model
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    metrics = evaluate_model(model, x_test, y_test, settings)

    # Print results to console (original behavior)
    print("Hotel booking cancellation classifier")
    print(f"Dataset:   {settings.data_path.name}")
    print(f"Target:    {settings.target_column}")
    print(f"Train/Test: {len(x_train)} / {len(x_test)} rows (seed {settings.random_seed})")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    log_mlflow_run(settings, model, metrics, y_test, y_pred, y_pred_proba)


if __name__ == "__main__":
    main()
