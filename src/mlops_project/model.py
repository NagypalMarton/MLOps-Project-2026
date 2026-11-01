from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings


def build_model(settings: Settings) -> Pipeline:
    """Build the logistic-regression cancellation classifier."""
    estimator = LogisticRegression(max_iter=settings.max_iter)
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def evaluate_model(model: Pipeline, x_test, y_test, settings: Settings) -> dict:
    predictions = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
    }
