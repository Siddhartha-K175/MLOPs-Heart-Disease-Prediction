import numpy as np
import pandas as pd
import pytest

from src import database


@pytest.fixture
def db_schema(monkeypatch):
    """Point every database helper at a clean, throwaway Postgres schema."""
    if not database.DB_URL:
        pytest.skip("SUPABASE_DB_URL is not set")
    monkeypatch.setattr(database, "DB_SCHEMA", database.TEST_SCHEMA)
    database.recreate_db(database.TEST_SCHEMA)
    yield database.TEST_SCHEMA


@pytest.fixture
def heart_frame() -> pd.DataFrame:
    """Small synthetic frame with the real schema and both target classes."""
    rng = np.random.default_rng(42)
    rows = 120
    frame = pd.DataFrame(
        {
            "age": rng.integers(30, 80, rows),
            "sex": rng.integers(0, 2, rows),
            "cp": rng.integers(1, 5, rows),
            "trestbps": rng.integers(100, 180, rows),
            "chol": rng.integers(150, 400, rows),
            "fbs": rng.integers(0, 2, rows),
            "restecg": rng.integers(0, 3, rows),
            "thalach": rng.integers(90, 200, rows),
            "exang": rng.integers(0, 2, rows),
            "oldpeak": rng.uniform(0, 4, rows).round(1),
            "slope": rng.integers(1, 4, rows),
            "ca": rng.integers(0, 4, rows),
            "thal": rng.integers(3, 8, rows),
        }
    )
    frame["target"] = rng.permutation(np.tile([0, 1], rows // 2))
    return frame
