from __future__ import annotations

import os
import joblib

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings, load_settings
from .model import build_model  # We can reuse the build_model function from model.py


def train_model(X_train, y_train, settings: Settings = None, model_output_path: str = "model/model.pkl") -> Pipeline:
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
    
    # Build the model (pipeline) using the existing function
    model = build_model(settings)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Save the model to disk
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    
    return model