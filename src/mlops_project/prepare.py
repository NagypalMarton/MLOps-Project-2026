from __future__ import annotations

import pandas as pd

from .config import Settings, load_settings
from .data import build_dataset


def prepare_data(settings: Settings | None = None) -> tuple:
    if settings is None:
        settings = load_settings()
    X_train, X_test, y_train, y_test = build_dataset(settings)
    save_prepared_data(X_train, X_test, y_train, y_test, "data/prepared.csv")
    return X_train, X_test, y_train, y_test


def save_prepared_data(X_train, X_test, y_train, y_test, filepath: str = "data/prepared.csv"):
    X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    X_train_df["target"] = y_train
    X_test_df["target"] = y_test
    X_train_df["split"] = "train"
    X_test_df["split"] = "test"
    combined_df = pd.concat([X_train_df, X_test_df], ignore_index=True)
    combined_df.to_csv(filepath, index=False)


if __name__ == "__main__":
    prepare_data()