from __future__ import annotations

import json
from pathlib import Path

from pandera.errors import SchemaError, SchemaErrors

from .config import load_settings
from .data import load_dataframe
from .schemas import validate_ingestion


REPORT_PATH = Path("reports/validation.json")


def _write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    settings = load_settings()
    frame = load_dataframe(settings)
    try:
        validate_ingestion(frame)
    except (SchemaError, SchemaErrors) as error:
        _write_report(
            {
                "valid": False,
                "rows": len(frame),
                "columns": len(frame.columns),
                "error": str(error),
            }
        )
        raise SystemExit("Data validation failed; see reports/validation.json.")

    _write_report(
        {
            "valid": True,
            "rows": len(frame),
            "columns": len(frame.columns),
            "contract": "Ingestion",
        }
    )


if __name__ == "__main__":
    main()
