from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import Settings, load_settings
from .data import _preprocess_features, build_dataset


def prepare_data(settings: Settings = None) -> tuple:
    """
    Prepare the dataset: load, preprocess features, and split.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if settings is None:
        settings = load_settings()
    
    # Build dataset (this loads, preprocesses, and splits)
    X_train, X_test, y_train, y_test = build_dataset(settings)
    
    # Save prepared data for DVC pipeline visibility
    save_prepared_data(X_train, X_test, y_train, y_test, "data/prepared.csv")
    
    return X_train, X_test, y_train, y_test


def save_prepared_data(X_train, X_test, y_train, y_test, filepath: str = "data/prepared.csv"):
    """
    Save prepared data to CSV for inspection/pipeline visibility.
    Note: For actual ML workflow, we'd typically save as numpy arrays or pickle,
    but for DVC visibility and simplicity, we save as CSV.
    """
    # Combine train and test data with a split indicator
    train_len = len(X_train)
    
    # Create dataframes
    X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    
    # Add target columns
    X_train_df['target'] = y_train
    X_test_df['target'] = y_test
    
    # Add split indicator
    X_train_df['split'] = 'train'
    X_test_df['split'] = 'test'
    
    # Combine and save
    combined_df = pd.concat([X_train_df, X_test_df], ignore_index=True)
    combined_df.to_csv(filepath, index=False)