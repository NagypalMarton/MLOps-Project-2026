"""Smoke tests that run WITHOUT your dataset (synthetic data), so the template is
green the moment you create it. Replace/extend these as your project grows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_project.config import Settings
from mlops_project.data import build_dataset
from mlops_project.model import build_model, evaluate_model


def _synthetic_classification(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "hotel": np.where(np.arange(n) % 2, "City Hotel", "Resort Hotel"),
            "is_canceled": target,
            "lead_time": rng.integers(0, 100, n),
            "arrival_date_year": 2016,
            "arrival_date_month": "July",
            "arrival_date_week_number": 28,
            "arrival_date_day_of_month": 15,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 1,
            "children": 0.0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "booking_changes": 0,
            "deposit_type": "No Deposit",
            "agent": 1.0,
            "company": 1.0,
            "days_in_waiting_list": 0,
            "customer_type": "Transient",
            "adr": 100.0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
            "reservation_status": "Check-Out",
            "reservation_status_date": "2016-07-15",
        }
    )
    return df


def test_classification_pipeline_trains():
    settings = Settings(test_size=0.25, random_seed=0)
    df = _synthetic_classification()
    x_tr, x_te, y_tr, y_te = build_dataset(settings, frame=df)
    model = build_model(settings)
    model.fit(x_tr, y_tr)
    metrics = evaluate_model(model, x_te, y_te, settings)
    assert {"accuracy", "precision", "recall", "f1"} <= set(metrics)
