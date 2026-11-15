from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path


DATA_PATH = Path("data/hotel_bookings.csv")
INVALID_LEAD_TIME = "999999"


def main() -> None:
    original = DATA_PATH.read_bytes()
    try:
        rows = list(csv.reader(original.decode("utf-8-sig").splitlines()))
        lead_time_index = rows[0].index("lead_time")
        rows[1][lead_time_index] = INVALID_LEAD_TIME
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        DATA_PATH.write_text(output.getvalue(), encoding="utf-8", newline="")
        print("Planted invalid lead_time=999999 in the first data row.")
        result = subprocess.run(["dvc", "repro", "validate"], check=False)
        print(f"dvc repro validate exited with code {result.returncode}.")
        if result.returncode == 0:
            raise SystemExit("The validation gate unexpectedly accepted bad data.")
    finally:
        DATA_PATH.write_bytes(original)
        print("Restored data/hotel_bookings.csv exactly.")


if __name__ == "__main__":
    main()
