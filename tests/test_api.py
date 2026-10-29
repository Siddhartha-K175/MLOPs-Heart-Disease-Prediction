import pytest
from fastapi.testclient import TestClient

import api
from src import auth
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


ADMIN_PASSWORD = "correct horse battery"


@pytest.fixture
def admin_headers(client):
    auth.create_admin("admin", ADMIN_PASSWORD)
    body = client.post(
        "/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


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


def test_predict_saves_record_and_patient_history(client, admin_headers):
    payload = {**VALID_PATIENT, "patient_id": "P-42", "name": "Asha Rao", "notes": "checkup"}
    body = client.post("/predict", json=payload).json()
    assert body["record_id"] is not None
    assert body["patient_id"] == "P-42"

    history = client.get("/patients/P-42/history", headers=admin_headers).json()
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


def test_predict_generates_id_when_only_name_is_given(client, admin_headers):
    body = client.post("/predict", json={**VALID_PATIENT, "name": "Ravi"}).json()
    assert body["patient_id"].startswith("P-")
    patient = client.get(f"/patients/{body['patient_id']}", headers=admin_headers).json()
    assert patient["name"] == "Ravi"


def test_predict_without_identity_stays_anonymous(client, admin_headers):
    body = client.post("/predict", json=VALID_PATIENT).json()
    assert body["patient_id"] is None
    assert body["record_id"] is not None
    recent = client.get("/predictions", headers=admin_headers).json()
    assert recent["count"] >= 1
    assert any(row["id"] == body["record_id"] for row in recent["predictions"])


def test_unknown_patient_returns_404(client, admin_headers):
    assert client.get("/patients/P-none", headers=admin_headers).status_code == 404
    response = client.get("/patients/P-none/history", headers=admin_headers)
    assert response.status_code == 404
    assert "P-none" in response.json()["detail"]


def test_patients_listing_counts_predictions(client, admin_headers):
    for _ in range(2):
        client.post("/predict", json={**VALID_PATIENT, "patient_id": "P-7", "name": "Ravi"})
    client.post("/predict", json=VALID_PATIENT)
    body = client.get("/patients", headers=admin_headers).json()
    assert body["count"] == 1
    assert body["patients"][0]["patient_id"] == "P-7"
    assert body["patients"][0]["prediction_count"] == 2
    assert body["patients"][0]["last_prediction_at"]


def test_identity_fields_are_trimmed(client, admin_headers):
    payload = {**VALID_PATIENT, "patient_id": " P-9 ", "name": "   "}
    body = client.post("/predict", json=payload).json()
    assert body["patient_id"] == "P-9"
    patient = client.get("/patients/P-9", headers=admin_headers).json()
    assert patient["name"] is None  # blank name treated as not provided


def test_recent_predictions_limit_is_validated(client, admin_headers):
    assert client.get("/predictions?limit=0", headers=admin_headers).status_code == 422
    assert client.get("/predictions?limit=999", headers=admin_headers).status_code == 422
    assert client.get("/predictions?limit=10", headers=admin_headers).status_code == 200


PROTECTED_ROUTES = ["/patients", "/patients/P-1", "/patients/P-1/history", "/predictions"]


@pytest.mark.parametrize("path", PROTECTED_ROUTES)
def test_patient_records_require_admin(client, path):
    response = client.get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("path", PROTECTED_ROUTES)
def test_patient_records_reject_invalid_token(client, path):
    response = client.get(path, headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_predict_stays_public_but_records_do_not(client):
    body = client.post("/predict", json={**VALID_PATIENT, "patient_id": "P-5"}).json()
    assert body["record_id"] is not None
    assert client.get("/patients/P-5/history").status_code == 401


def test_login_rejects_wrong_password_and_unknown_user(client):
    auth.create_admin("admin", ADMIN_PASSWORD)
    wrong = client.post("/auth/login", json={"username": "admin", "password": "nope-nope-nope"})
    unknown = client.post("/auth/login", json={"username": "ghost", "password": ADMIN_PASSWORD})
    assert wrong.status_code == 401
    assert unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_me_and_logout_end_the_session(client, admin_headers):
    me = client.get("/auth/me", headers=admin_headers).json()
    assert me == {"username": "admin", "role": "admin", "patient_id": None, "name": None}
    assert client.post("/auth/logout", headers=admin_headers).status_code == 200
    assert client.get("/patients", headers=admin_headers).status_code == 401


def _register(client, username="asha", name="Asha Rao"):
    response = client.post(
        "/auth/register", json={"username": username, "password": ADMIN_PASSWORD, "name": name}
    )
    assert response.status_code == 201
    body = response.json()
    return body, {"Authorization": f"Bearer {body['access_token']}"}


def test_register_signs_patient_in(client):
    body, headers = _register(client)
    assert body["role"] == "patient"
    assert body["patient_id"].startswith("P-")
    me = client.get("/auth/me", headers=headers).json()
    assert me == {
        "username": "asha", "role": "patient", "patient_id": body["patient_id"],
        "name": "Asha Rao",
    }


def test_register_rejects_duplicates_and_weak_input(client):
    _register(client)
    duplicate = client.post(
        "/auth/register", json={"username": "ASHA", "password": ADMIN_PASSWORD, "name": "X"}
    )
    assert duplicate.status_code == 409
    weak = client.post(
        "/auth/register", json={"username": "ravi", "password": "short", "name": "R"}
    )
    assert weak.status_code == 422


def test_patient_assessments_go_to_their_own_record(client):
    body, headers = _register(client)
    saved = client.post(
        "/predict", json={**VALID_PATIENT, "patient_id": "P-SOMEONE-ELSE", "name": "Mallory"},
        headers=headers,
    ).json()
    assert saved["patient_id"] == body["patient_id"]
    history = client.get("/me/history", headers=headers).json()
    assert history["count"] == 1
    assert history["patient"]["name"] == "Asha Rao"  # body identity ignored


def test_patient_sees_only_own_history(client, admin_headers):
    client.post("/predict", json={**VALID_PATIENT, "patient_id": "P-OTHER", "name": "Ravi"})
    _, headers = _register(client)
    assert client.get("/me/history", headers=headers).json()["count"] == 0
    for path in PROTECTED_ROUTES:
        assert client.get(path, headers=headers).status_code == 403
    assert client.get("/me/history", headers=admin_headers).status_code == 403
    assert client.get("/me/history").status_code == 401


def test_invalid_token_on_predict_is_rejected(client):
    response = client.post(
        "/predict", json=VALID_PATIENT, headers={"Authorization": "Bearer stale-token"}
    )
    assert response.status_code == 401
