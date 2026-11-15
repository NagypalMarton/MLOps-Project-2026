from __future__ import annotations

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .config import Settings, load_settings
from .data import build_dataset, load_dataframe


def prepare_data(settings: Settings | None = None) -> tuple:
    if settings is None:
        settings = load_settings()
    X_train, X_test, y_train, y_test = build_dataset(settings)
    raw = load_dataframe(settings)
    guest_total = (
        raw["adults"] + raw["children"].fillna(0) + raw["babies"]
    )
    raw = raw.loc[guest_total > 0].reset_index(drop=True)
    with open("params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}
    slice_columns = [
        definition["column"] for definition in params.get("slices", [])
    ]
    raw_train, raw_test = train_test_split(
        raw[slice_columns],
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=raw["is_canceled"],
    )
    save_prepared_data(
        X_train,
        X_test,
        y_train,
        y_test,
        "data/prepared.csv",
        raw_train.reset_index(drop=True),
        raw_test.reset_index(drop=True),
    )
    return X_train, X_test, y_train, y_test


def save_prepared_data(
    X_train,
    X_test,
    y_train,
    y_test,
    filepath: str = "data/prepared.csv",
    raw_train: pd.DataFrame | None = None,
    raw_test: pd.DataFrame | None = None,
):
    X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    X_train_df["target"] = y_train
    X_test_df["target"] = y_test
    X_train_df["split"] = "train"
    X_test_df["split"] = "test"
    if raw_train is not None and raw_test is not None:
        for column in raw_train.columns:
            X_train_df[f"raw__{column}"] = raw_train[column].to_numpy()
            X_test_df[f"raw__{column}"] = raw_test[column].to_numpy()
    combined_df = pd.concat([X_train_df, X_test_df], ignore_index=True)
    combined_df.to_csv(filepath, index=False)


if __name__ == "__main__":
    prepare_data()