from __future__ import annotations

import os
import joblib
import pandas as pd
import yaml
from dataclasses import replace

from sklearn.pipeline import Pipeline

from .config import Settings, load_settings
from .model import build_model, evaluate_model
from .cli import log_mlflow_run


def train_model(
    X_train,
    y_train,
    settings: Settings | None = None,
    model_output_path: str = "model/model.pkl",
) -> Pipeline:
    """
    Train the model and save it to disk.
    
    Args:
        X_train: Training features
        y_train: Training target
        settings: Settings object (optional, will load if not provided)
        model_output_path: Path to save the trained model
    
    Returns:
        Pipeline: Trained scikit-learn pipeline
    """
    if settings is None:
        settings = load_settings()
    if os.path.exists("params.yaml"):
        with open("params.yaml", encoding="utf-8") as file:
            params = yaml.safe_load(file) or {}
        max_iter = params.get("model", {}).get("max_iter")
        if max_iter is not None:
            settings = replace(settings, max_iter=int(max_iter))
    
    # Build the model (pipeline) using the existing function
    model = build_model(settings)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Save the model to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    
    return model


def _load_prepared_data(filepath: str = "data/prepared.csv"):
    prepared = pd.read_csv(filepath)
    train = prepared[prepared["split"] == "train"]
    feature_columns = [column for column in prepared.columns if column.startswith("feature_")]
    test = prepared[prepared["split"] == "test"]
    return (
        train[feature_columns].to_numpy(),
        train["target"].to_numpy(),
        test[feature_columns].to_numpy(),
        test["target"].to_numpy(),
    )


if __name__ == "__main__":
    settings = load_settings()
    X_train, y_train, X_test, y_test = _load_prepared_data()
    model = train_model(X_train, y_train, settings)
    metrics = evaluate_model(model, X_test, y_test, settings)
    predictions = model.predict(X_test)
    probabilities = (
        model.predict_proba(X_test)[:, 1]
        if settings.problem_type == "classification"
        else None
    )
    log_mlflow_run(settings, model, metrics, y_test, predictions, probabilities)