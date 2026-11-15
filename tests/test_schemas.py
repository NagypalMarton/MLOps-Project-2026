import pytest
from pandera.errors import SchemaError, SchemaErrors

from mlops_project.data import _clean_for_model
from mlops_project.schemas import Ingestion, ModelInput
from tests.test_smoke import _synthetic_classification


def test_ingestion_contract_is_strict():
    frame = _synthetic_classification(8)
    frame["unexpected_column"] = 1

    with pytest.raises(SchemaErrors):
        Ingestion.validate(frame)


def test_model_input_rejects_negative_adr():
    frame = _clean_for_model(_synthetic_classification(8))
    frame.loc[0, "adr"] = -1.0

    with pytest.raises(SchemaError):
        ModelInput.validate(frame)


def test_model_input_rejects_zero_guest_rows():
    frame = _clean_for_model(_synthetic_classification(8))
    frame.loc[0, "adults"] = 0
    frame.loc[0, "children"] = 0.0
    frame.loc[0, "babies"] = 0
    frame.loc[0, "total_guests"] = 0.0

    with pytest.raises(SchemaError):
        ModelInput.validate(frame)
