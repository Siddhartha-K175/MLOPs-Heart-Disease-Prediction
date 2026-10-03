import pytest
from fastapi.testclient import TestClient

import api
from src.train import train

VALID_PATIENT = {
    "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 240, "fbs": 0,
    "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1,
    "ca": 0, "thal": 2,
}


@pytest.fixture
def client(tmp_path, heart_frame, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    train(heart_frame, artifact_dir, log_to_mlflow=False)
    monkeypatch.setattr(api, "ARTIFACT", artifact_dir / "heart_model.joblib")
    with TestClient(api.app) as test_client:
        yield test_client


def test_home_lists_endpoints(client):
    body = client.get("/").json()
    assert body["service"] == "Heart disease prediction API"
    assert "/predict" in body["endpoints"]


def test_health_reports_training_metrics(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "accuracy" in body
    assert "data_hash" in body


def test_predict_returns_probability_and_version(client):
    body = client.post("/predict", json=VALID_PATIENT).json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_label"]
    assert body["model_version"] is None  # trained without MLflow in tests


def test_predict_rejects_out_of_range_input(client):
    payload = {**VALID_PATIENT, "age": 250}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch(client):
    response = client.post("/predict-batch", json=[VALID_PATIENT, VALID_PATIENT])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_predict_batch_rejects_empty_list(client):
    assert client.post("/predict-batch", json=[]).status_code == 422


def test_predict_batch_enforces_row_limit(client):
    oversized = [VALID_PATIENT] * (api.MAX_BATCH_ROWS + 1)
    response = client.post("/predict-batch", json=oversized)
    assert response.status_code == 422
    assert "limit" in response.json()["detail"]


def test_predict_csv_returns_csv_with_predictions(client, heart_frame):
    payload = heart_frame.head(5).to_csv(index=False).encode()
    response = client.post(
        "/predict-csv", files={"file": ("patients.csv", payload, "text/csv")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.strip().splitlines()
    assert len(lines) == 6
    assert "risk_probability" in lines[0]


def test_predict_csv_rejects_wrong_columns(client):
    payload = b"unrelated,columns\n1,2\n"
    response = client.post("/predict-csv", files={"file": ("bad.csv", payload, "text/csv")})
    assert response.status_code == 422


def test_predict_csv_rejects_garbage_bytes(client):
    payload = b"\x00\x01\x02 not a csv"
    response = client.post("/predict-csv", files={"file": ("junk.csv", payload, "text/csv")})
    assert response.status_code == 422
