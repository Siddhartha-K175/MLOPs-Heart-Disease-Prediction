"""Tests for the db.py command-line SQL runner."""

import pytest

from db import main, run_sql
from src import database

FEATURES_INPUT = {
    "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 240, "fbs": 0,
    "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1,
    "ca": 0, "thal": 2,
}


@pytest.fixture
def seed(db_schema):
    database.save_prediction(
        features=FEATURES_INPUT, prediction=0,
        risk_probability=0.2, risk_label="Lower likelihood of heart disease",
        patient_id="P-1", name="Asha",
    )
    return db_schema


def test_run_sql_select(seed):
    connection = database.connect()
    columns, rows, count = run_sql(connection, "SELECT patient_id, name FROM patients")
    connection.close()
    assert columns == ["patient_id", "name"]
    assert rows == [("P-1", "Asha")]
    assert count == 1


def test_run_sql_mutation_commits(seed):
    connection = database.connect()
    columns, rows, count = run_sql(
        connection, "UPDATE patients SET notes = 'reviewed' WHERE patient_id = 'P-1'"
    )
    connection.close()
    assert columns is None and rows == []
    assert count == 1
    assert database.get_patient("P-1")["notes"] == "reviewed"


def test_cli_prints_query_results(seed, capsys):
    assert main(["SELECT patient_id FROM patients;"]) == 0
    output = capsys.readouterr().out
    assert "patient_id" in output
    assert "P-1" in output
    assert "(1 row(s))" in output


def test_cli_reports_affected_rows(seed, capsys):
    assert main(["DELETE FROM predictions;"]) == 0
    assert "1 row(s) affected." in capsys.readouterr().out


def test_cli_creates_a_missing_schema(db_schema, capsys):
    fresh = "heart_health_cli_fresh"
    database.drop_db(fresh)
    try:
        assert main(["--schema", fresh, "SELECT count(*) FROM predictions;"]) == 0
        out = capsys.readouterr().out
        assert f"Created and initialized schema `{fresh}`" in out
        assert "(1 row(s))" in out
    finally:
        database.drop_db(fresh)


def test_cli_returns_error_for_bad_sql(seed, capsys):
    assert main(["SELEC nonsense"]) == 1
    assert "SQL error" in capsys.readouterr().err
