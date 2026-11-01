from __future__ import annotations

import json
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

import yaml

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
        return "unknown"
    with open(dvc_file, "r") as f:
        meta = yaml.safe_load(f)
    return meta["outs"][0]["md5"]


def _log_classification_plots(y_true, y_pred, y_pred_proba, prefix="") -> None:
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


def main() -> None:
    # Load settings
    settings = load_settings()

    # Load and split data
    x_train, x_test, y_train, y_test = build_dataset(settings)

    # Build and train model
    model = build_model(settings)
    model.fit(x_train, y_train)

    # Evaluate model
    if settings.problem_type == "classification":
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]  # probability of positive class
    else:
        y_pred = model.predict(x_test)
        y_pred_proba = None

    metrics = evaluate_model(model, x_test, y_test, settings)

    # Print results to console (original behavior)
    print(f"Project baseline — {settings.problem_type}")
    print(f"Dataset:   {settings.data_path.name}")
    print(f"Target:    {settings.target_column}")
    print(f"Train/Test: {len(x_train)} / {len(x_test)} rows (seed {settings.random_seed})")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    # MLflow integration - only attempt if tracking URI is set
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri and tracking_uri.strip():
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("hotel-cancel-prediction")
            with mlflow.start_run() as run:
                # Log parameters
                _log_settings_as_params(settings)
                _log_params_yaml_as_params("params.yaml")
                dataset_version = _get_dataset_version()
                mlflow.log_param("dataset_version", dataset_version)

                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Log model
                mlflow.sklearn.log_model(model, "model")

                # For classification, log plots
                if settings.problem_type == "classification":
                    _log_classification_plots(y_test, y_pred, y_pred_proba)

                # Register model and update alias if it's the best
                model_name = "hotel-cancel-predictor"
                client = MlflowClient()
                try:
                    client.get_registered_model(model_name)
                except RestException:
                    client.create_registered_model(model_name)

                # Create a new version of the model from this run
                model_uri = f"runs://{run.info.run_id}/model"
                version = client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run.info.run_id
                )

                # Find the best version based on F1-score (for classification) or R2 (for regression)
                versions = client.get_model_versions(filter_string=f"name='{model_name}'")
                best_version = None
                best_metric = -1.0
                metric_name = "f1" if settings.problem_type == "classification" else "r2"
                for v in versions:
                    v_run = client.get_run(v.run_id)
                    metric_value = v_run.data.metrics.get(metric_name, 0.0)
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_version = v.version

                # If this version is the best, set the "staging" alias
                if version.version == best_version:
                    client.set_registered_model_alias(model_name, "staging", version.version)
        except Exception as e:
            # If any MLstep fails, we just print a warning and continue without MLflow
            print(f"Warning: MLflow logging failed: {e}. Continuing without MLflow.")
    else:
        # No tracking URI set, skip MLflow entirely
        print("Info: MLFLOW_TRACKING_URI not set, skipping MLflow.")


if __name__ == "__main__":
    main()
