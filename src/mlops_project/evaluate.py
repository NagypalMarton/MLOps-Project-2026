from __future__ import annotations

import json
import os
import joblib
from pathlib import Path
import pandas as pd
import yaml

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
    _save_slice_metrics(test, x_test, y_test, model, settings, metrics_path)
    _save_fairness_metrics(test, x_test, y_test, model)
    
    return metrics


def _save_slice_metrics(
    test: pd.DataFrame,
    x_test,
    y_test,
    model,
    settings,
    metrics_path: str,
) -> None:
    with open("params.yaml", encoding="utf-8") as file:
        definitions = (yaml.safe_load(file) or {}).get("slices", [])

    rows = []
    for definition in definitions:
        column = definition["column"]
        raw_column = f"raw__{column}"
        for value in definition["values"]:
            mask = test[raw_column].eq(value).to_numpy()
            slice_y = y_test[mask]
            row = {
                "slice": definition["name"],
                "column": column,
                "value": value,
                "n": int(mask.sum()),
                "positive_labels": int(slice_y.sum()),
            }
            if row["n"]:
                row.update(evaluate_model(model, x_test[mask], slice_y, settings))
            else:
                row.update(
                    {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                    }
                )
            rows.append(row)

    slices_path = Path(metrics_path).with_name("slices.json")
    slices_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _save_fairness_metrics(
    test: pd.DataFrame,
    x_test,
    y_test,
    model,
    output_path: str = "metrics/fairness.json",
) -> None:
    with open("params.yaml", encoding="utf-8") as file:
        fairness = (yaml.safe_load(file) or {}).get("fairness", {})
    slice_name = fairness["slice"]
    definition = next(
        definition
        for definition in (yaml.safe_load(Path("params.yaml").read_text()) or {}).get(
            "slices", []
        )
        if definition["name"] == slice_name
    )
    predictions = model.predict(x_test)
    groups = []
    for value in definition["values"]:
        mask = test[f"raw__{definition['column']}"].eq(value).to_numpy()
        group_y = y_test[mask]
        group_predictions = predictions[mask]
        n = int(mask.sum())
        groups.append(
            {
                "value": value,
                "n": n,
                "positive_labels": int(group_y.sum()),
                "predicted_positive": int(group_predictions.sum()),
                "selection_rate": (
                    round(float(group_predictions.mean()), 4) if n else None
                ),
            }
        )

    min_support = int(fairness["min_support"])
    eligible = [
        group
        for group in groups
        if group["n"] >= min_support and group["selection_rate"] is not None
    ]
    rates = [group["selection_rate"] for group in eligible]
    report = {
        "slice": slice_name,
        "column": definition["column"],
        "metric": fairness["metric"],
        "min_support": min_support,
        "groups": groups,
        "eligible_groups": [group["value"] for group in eligible],
        "selection_rate_gap": round(max(rates) - min(rates), 4) if rates else None,
    }
    Path(output_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    evaluate_and_save_metrics()