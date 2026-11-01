import pytest

from mlops_project.config import Settings, _validate


def test_defaults():
    s = Settings()
    assert s.target_column == "is_canceled"
    assert 0.0 < s.test_size < 1.0


def test_invalid_test_size_rejected():
    with pytest.raises(ValueError):
        _validate(Settings(test_size=1.5))
