from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PARAMS = Path("params.yaml")
MODEL = Path("model/model.pkl")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def main() -> None:
    original = PARAMS.read_text(encoding="utf-8")
    try:
        PARAMS.write_text(original.replace("max_iter: 1000", "max_iter: 1"), encoding="utf-8")
        repro = run(["uv", "run", "dvc", "repro"])
        print(repro.stdout)
        print(repro.stderr)
        print(f"dvc_repro_exit={repro.returncode}")
        if repro.returncode == 0:
            raise RuntimeError("The deliberately weak model unexpectedly passed the gate.")
        if not MODEL.exists():
            raise RuntimeError("The failed model artifact was unexpectedly removed.")
        publish = run(["uv", "run", "python", "-m", "mlops_project.publish"])
        print(publish.stdout)
        print(publish.stderr)
        print(f"publish_exit={publish.returncode}")
        if publish.returncode == 0:
            raise RuntimeError("Publication unexpectedly succeeded for a NO-GO verdict.")
        print(f"model_exists={MODEL.exists()}")
    finally:
        PARAMS.write_text(original, encoding="utf-8")
        restored = run(["uv", "run", "dvc", "repro"])
        print(restored.stdout)
        print(restored.stderr)
        print(f"restored_dvc_repro_exit={restored.returncode}")
        if restored.returncode != 0:
            raise RuntimeError("Restoring the GO configuration failed.")


if __name__ == "__main__":
    main()
