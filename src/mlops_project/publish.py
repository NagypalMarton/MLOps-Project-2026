from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

VERDICT_PATH = Path("reports/gate.json")
RUN_PATH = Path("reports/latest_run.json")
MODEL_NAME = "hotel-cancel-predictor"
ALIAS = "staging"


def publish_model(
    verdict_path: Path = VERDICT_PATH,
    run_path: Path = RUN_PATH,
) -> str:
    load_dotenv()
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict.get("status") != "pass":
        raise RuntimeError(
            "Model publication refused: reports/gate.json is not a GO verdict "
            f"(status={verdict.get('status')!r})."
        )

    run_id = json.loads(run_path.read_text(encoding="utf-8"))["run_id"]
    client = MlflowClient()
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(MODEL_NAME)

    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'"
    )
    version = next((item for item in versions if item.run_id == run_id), None)
    if version is None:
        version = client.create_model_version(
            name=MODEL_NAME,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    print(
        f"published model={MODEL_NAME} version={version.version} "
        f"alias={ALIAS} run_id={run_id}"
    )
    return str(version.version)


def main() -> None:
    try:
        publish_model()
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        print(str(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
