import pandas as pd
import pytest

from src.model import FEATURES, range_violations, validate_features


def test_validate_features_orders_columns_and_converts_numbers():
    source = pd.DataFrame([{feature: "1" for feature in reversed(FEATURES)}])
    result = validate_features(source)
    assert result.columns.tolist() == FEATURES
    assert result.dtypes.astype(str).eq("int64").all()


def test_validate_features_reports_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_features(pd.DataFrame({"age": [55]}))


def test_range_violations_accepts_valid_frame():
    valid = {
        "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 240, "fbs": 0,
        "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1,
        "ca": 0, "thal": 2,
    }
    assert range_violations(pd.DataFrame([valid])) == {}


def test_range_violations_flags_bad_values():
    frame = pd.DataFrame([{"age": 250, "chol": 40, "thalach": 500}])
    violations = range_violations(frame)
    assert set(violations) == {"age", "chol", "thalach"}
