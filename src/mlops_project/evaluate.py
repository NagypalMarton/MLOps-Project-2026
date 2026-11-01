from __future__ import annotations

import json
import os
import joblib
import pandas as pd

from .config import load_settings
from .model import evaluate_model


def evaluate_and_save_metrics(
    model_input_path: str = "model/model.pkl",
    metrics_path: str = "metrics/metrics.json",
) -> dict:
    settings = load_settings()
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    prepared = pd.read_csv("data/prepared.csv")
    test = prepared[prepared["split"] == "test"]
    feature_columns = [column for column in prepared.columns if column.startswith("feature_")]
    x_test = test[feature_columns].to_numpy()
    y_test = test["target"].to_numpy()
    
    model = joblib.load(model_input_path)
    metrics = evaluate_model(model, x_test, y_test, settings)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    evaluate_and_save_metrics()