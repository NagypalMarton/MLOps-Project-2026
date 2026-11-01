from __future__ import annotations

import json
import os
import joblib
import pandas as pd

from .config import load_settings
from .model import evaluate_model


def evaluate_and_save_metrics(model_input_path: str = "model/model.pkl", metrics_path: str = "metrics/metrics.json"):
    """
    Load the trained model, evaluate it, and save metrics to JSON file.
    
    Args:
        model_input_path: Path to the saved model
        metrics_path: Path to save metrics JSON
    
    Returns:
        dict: Evaluation metrics
    """
    # Load settings
    settings = load_settings()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    # Load the prepared test split
    prepared = pd.read_csv("data/prepared.csv")
    test = prepared[prepared["split"] == "test"]
    feature_columns = [column for column in prepared.columns if column.startswith("feature_")]
    x_test = test[feature_columns].to_numpy()
    y_test = test["target"].to_numpy()
    
    # Load the trained model
    model = joblib.load(model_input_path)
    
    # Evaluate the model
    metrics = evaluate_model(model, x_test, y_test, settings)
    
    # Save metrics to JSON
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    evaluate_and_save_metrics()