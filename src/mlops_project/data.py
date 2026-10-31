from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import Settings


def load_dataframe(settings: Settings) -> pd.DataFrame:
    """Load your dataset CSV (set DATA_PATH in .env)."""
    if not settings.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {settings.data_path}\n"
            "Place your dataset under data/ and set DATA_PATH in .env."
        )
    return pd.read_csv(settings.data_path)


def _preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess features to a numeric numpy array.
    - Drops known leakage columns.
    - Separates numeric and categorical columns.
    - Numeric: fill missing with median.
    - Categorical: fill missing with 'missing' then label encode.
    Returns a 2D numpy array (n_samples, n_features).
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Drop leakage columns if present
    leakage_columns = ["reservation_status", "reservation_status_date"]
    for col in leakage_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Separate feature types
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns

    processed_arrays = []

    # Process numeric columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        processed_arrays.append(df[col].values.reshape(-1, 1))

    # Process categorical columns
    for col in categorical_cols:
        df[col] = df[col].fillna("missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        processed_arrays.append(df[col].values.reshape(-1, 1))

    # Horizontally stack all processed columns
    if processed_arrays:
        return np.hstack(processed_arrays)
    else:
        # Return empty array with correct number of rows
        return np.empty((len(df), 0))


def build_dataset(settings: Settings, frame: pd.DataFrame | None = None):
    """Split into train/test. Stratifies on the target for classification."""
    frame = load_dataframe(settings) if frame is None else frame
    if settings.target_column not in frame.columns:
        raise KeyError(
            f"Target column '{settings.target_column}' not in data "
            f"(columns: {list(frame.columns)}). Set TARGET_COLUMN in .env."
        )
    # Separate features and target
    y = frame[settings.target_column]
    X = frame.drop(columns=[settings.target_column])
    # Preprocess features to numeric array
    X_processed = _preprocess_features(X)
    labels = y.values
    stratify = labels if settings.problem_type == "classification" else None
    return train_test_split(
        X_processed,
        labels,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=stratify,
    )
