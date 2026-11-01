from __future__ import annotations

import json
import os
import joblib

from .config import Settings, load_settings
from .data import build_dataset
from .model import build_model, evaluate_model


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
    
    # Load and split data
    _, x_test, _, y_test = build_dataset(settings)
    
    # Load the trained model
    model = joblib.load(model_input_path)
    
    # Evaluate the model
    metrics = evaluate_model(model, x_test, y_test, settings)
    
    # Save metrics to JSON
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics