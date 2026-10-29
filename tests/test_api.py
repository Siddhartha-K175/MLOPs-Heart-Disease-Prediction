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
def client(tmp_path, heart_frame, monkeypatch, db_schema):
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


def test_predict_includes_prevention_guidance(client):
    body = client.post("/predict", json=VALID_PATIENT).json()
    guidance = body["guidance"]
    assert guidance is not None
    assert guidance["tier"] in {"low", "moderate", "high"}
    assert guidance["headline"]
    assert guidance["actions"]
    assert "Seek emergency care" in guidance["emergency_note"]
    assert "clinician" in guidance["disclaimer"]


def test_predict_batch_keeps_responses_lean(client):
    response = client.post("/predict-batch", json=[VALID_PATIENT])
    assert response.json()[0]["guidance"] is None


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


def test_predict_saves_record_and_patient_history(client):
    payload = {**VALID_PATIENT, "patient_id": "P-42", "name": "Asha Rao", "notes": "checkup"}
    body = client.post("/predict", json=payload).json()
    assert body["record_id"] is not None
    assert body["patient_id"] == "P-42"

    history = client.get("/patients/P-42/history").json()
    assert history["count"] == 1
    assert history["patient"]["name"] == "Asha Rao"
    assert history["patient"]["notes"] == "checkup"
    saved = history["history"][0]
    assert saved["age"] == 55
    assert saved["chol"] == 240
    assert saved["record_id"] == body["record_id"]
    assert saved["prediction"] == body["prediction"]
    assert saved["risk_probability"] == pytest.approx(body["risk_probability"])
    assert saved["created_at"]


def test_predict_generates_id_when_only_name_is_given(client):
    body = client.post("/predict", json={**VALID_PATIENT, "name": "Ravi"}).json()
    assert body["patient_id"].startswith("P-")
    assert client.get(f"/patients/{body['patient_id']}").json()["name"] == "Ravi"


def test_predict_without_identity_stays_anonymous(client):
    body = client.post("/predict", json=VALID_PATIENT).json()
    assert body["patient_id"] is None
    assert body["record_id"] is not None
    recent = client.get("/predictions").json()
    assert recent["count"] >= 1
    assert any(row["id"] == body["record_id"] for row in recent["predictions"])


def test_unknown_patient_returns_404(client):
    assert client.get("/patients/P-none").status_code == 404
    response = client.get("/patients/P-none/history")
    assert response.status_code == 404
    assert "P-none" in response.json()["detail"]


def test_patients_listing_counts_predictions(client):
    for _ in range(2):
        client.post("/predict", json={**VALID_PATIENT, "patient_id": "P-7", "name": "Ravi"})
    client.post("/predict", json=VALID_PATIENT)
    body = client.get("/patients").json()
    assert body["count"] == 1
    assert body["patients"][0]["patient_id"] == "P-7"
    assert body["patients"][0]["prediction_count"] == 2
    assert body["patients"][0]["last_prediction_at"]


def test_identity_fields_are_trimmed(client):
    payload = {**VALID_PATIENT, "patient_id": " P-9 ", "name": "   "}
    body = client.post("/predict", json=payload).json()
    assert body["patient_id"] == "P-9"
    patient = client.get("/patients/P-9").json()
    assert patient["name"] is None  # blank name treated as not provided


def test_recent_predictions_limit_is_validated(client):
    assert client.get("/predictions?limit=0").status_code == 422
    assert client.get("/predictions?limit=999").status_code == 422
    assert client.get("/predictions?limit=10").status_code == 200
