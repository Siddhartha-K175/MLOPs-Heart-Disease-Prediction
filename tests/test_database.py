"""Unit tests for the Supabase/Postgres persistence layer (src/database.py + schema.sql)."""

import pytest

from src import database

FEATURES_INPUT = {
    "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 240, "fbs": 0,
    "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1,
    "ca": 0, "thal": 2,
}


def _save(**overrides):
    params = {
        "features": FEATURES_INPUT,
        "prediction": 1,
        "risk_probability": 0.8,
        "risk_label": "Higher likelihood of heart disease",
        "model_version": "v1",
    }
    params.update(overrides)
    return database.save_prediction(**params)


def test_init_db_is_idempotent(db_schema):
    database.init_db()  # second run must not fail or duplicate
    database.init_db()
    connection = database.connect()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name AS object_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        )
        tables = {row["object_name"] for row in cursor.fetchall()}
        cursor.execute(
            "SELECT table_name AS object_name FROM information_schema.views "
            "WHERE table_schema = current_schema()"
        )
        views = {row["object_name"] for row in cursor.fetchall()}
    connection.close()
    assert {"patients", "predictions"} <= tables
    assert "patient_history" in views


def test_round_trip_stores_inputs_and_result(db_schema):
    saved = _save(patient_id="P-1", name="Asha", notes="family history")
    assert saved == {"record_id": 1, "patient_id": "P-1"}

    record = database.get_patient_with_history("P-1")
    assert record["patient"]["name"] == "Asha"
    assert record["patient"]["notes"] == "family history"
    assert record["count"] == 1
    row = record["history"][0]
    assert row["age"] == 55
    assert row["chol"] == 240
    assert row["oldpeak"] == 1.0
    assert row["prediction"] == 1
    assert row["risk_probability"] == pytest.approx(0.8)
    assert row["risk_label"] == "Higher likelihood of heart disease"
    assert row["model_version"] == "v1"
    assert row["created_at"]


def test_generates_id_when_only_name_is_given(db_schema):
    saved = _save(name="Ravi")
    assert saved["patient_id"].startswith("P-")
    assert database.get_patient(saved["patient_id"])["name"] == "Ravi"


def test_anonymous_prediction_kept_without_patient(db_schema):
    saved = _save()
    assert saved["patient_id"] is None
    assert database.list_patients() == []
    assert database.get_patient_with_history("P-1") is None
    recent = database.recent_predictions()
    assert len(recent) == 1
    assert recent[0]["patient_id"] is None
    assert recent[0]["age"] == 55


def test_repeat_predictions_append_history_and_update_name(db_schema):
    _save(patient_id="P-2", name="Ravi")
    _save(patient_id="P-2", name="Ravi Kumar")
    record = database.get_patient_with_history("P-2")
    assert record["count"] == 2
    assert record["patient"]["name"] == "Ravi Kumar"
    patients = database.list_patients()
    assert len(patients) == 1
    assert patients[0]["prediction_count"] == 2
    assert patients[0]["latest_risk"] == pytest.approx(0.8)


def test_existing_name_survives_a_resave_without_one(db_schema):
    _save(patient_id="P-3", name="Asha")
    _save(patient_id="P-3")
    assert database.get_patient("P-3")["name"] == "Asha"


def test_missing_feature_is_rejected_before_sql():
    with pytest.raises(ValueError, match="Missing features"):
        _save(features={"age": 55})


def test_schema_check_constraints_reject_out_of_range_values(db_schema):
    with pytest.raises(database.DatabaseError):
        _save(features={**FEATURES_INPUT, "age": 250})
    with pytest.raises(database.DatabaseError):
        _save(prediction=7)


def test_unknown_patient_returns_none(db_schema):
    assert database.get_patient("P-missing") is None
    assert database.get_patient_with_history("P-missing") is None


def test_history_is_newest_first(db_schema):
    _save(patient_id="P-4", name="Meera")
    _save(patient_id="P-4", name="Meera")
    recent = database.recent_predictions(limit=1)
    assert recent[0]["id"] == 2
    history = database.get_history("P-4")
    assert [row["record_id"] for row in history] == [2, 1]
