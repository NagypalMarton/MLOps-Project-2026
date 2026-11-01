from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    """Configuration for hotel booking cancellation classification."""

    data_path: Path = Path("data/hotel_bookings.csv")
    target_column: str = "is_canceled"
    test_size: float = 0.25
    random_seed: int = 42
    max_iter: int = 1000


def load_settings(project_root: Path | None = None) -> Settings:
    base_path = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(base_path / ".env")

    settings = Settings(
        data_path=base_path / os.getenv("DATA_PATH", "data/hotel_bookings.csv"),
        target_column=os.getenv("TARGET_COLUMN", "is_canceled"),
        test_size=float(os.getenv("TEST_SIZE", "0.25")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        max_iter=1000,
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if not 0.0 < settings.test_size < 1.0:
        raise ValueError("TEST_SIZE must be between 0 and 1 (exclusive).")
    if settings.max_iter <= 0:
        raise ValueError("MAX_ITER must be greater than 0.")
