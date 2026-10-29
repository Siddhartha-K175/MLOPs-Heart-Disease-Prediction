"""FastAPI service exposing the heart-disease model.

Usage:
    uv run uvicorn api:app --reload --port 8000
    Open http://127.0.0.1:8000/docs for the interactive API reference.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, create_model

from src import auth, database
from src.advice import recommendations
from src.model import (
    FEATURE_BOUNDS,
    FEATURE_DEFAULTS,
    FEATURE_LABELS,
    FEATURES,
    load_bundle,
    predict,
)

LOGGER = logging.getLogger("heart.api")
ROOT = Path(__file__).parent
ARTIFACT = ROOT / "artifacts" / "heart_model.joblib"
MAX_BATCH_ROWS = 1000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
IDENTITY_FIELDS = ("patient_id", "name", "notes")

Patient = create_model(
    "Patient",
    **{
        feature: (
            float,
            Field(
                float(FEATURE_DEFAULTS[feature]),
                ge=FEATURE_BOUNDS[feature][0],
                le=FEATURE_BOUNDS[feature][1],
                description=FEATURE_LABELS[feature],
            ),
        )
        for feature in FEATURES
    },
    # Optional identity used to link the prediction to a patient history row.
    patient_id=(
        str | None,
        Field(None, max_length=64, description="Client-assigned patient id (e.g. P-1001)."),
    ),
    name=(
        str | None,
        Field(
            None,
            max_length=120,
            description="Patient name; an id is generated when patient_id is empty.",
        ),
    ),
    notes=(
        str | None,
        Field(None, max_length=500, description="Free-text clinical note stored with the patient."),
    ),
)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="Letters, digits, . - _")
    password: str = Field(..., min_length=auth.MIN_PASSWORD_LENGTH, max_length=256)
    name: str = Field(..., min_length=1, max_length=120, description="Patient's full name")


class Session(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str = Field(..., description="'admin' or 'patient'")
    patient_id: str | None = Field(None, description="Linked patient record (patients only)")
    expires_at: datetime


class Guidance(BaseModel):
    tier: str
    headline: str
    actions: list[str]
    emergency_note: str
    disclaimer: str


class Prediction(BaseModel):
    prediction: int
    risk_probability: float
    risk_label: str
    model_version: str | None = None
    guidance: Guidance | None = None
    record_id: int | None = Field(None, description="Database id of the saved prediction row.")
    patient_id: str | None = Field(None, description="Patient id the record was linked to.")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the trained model bundle and create the database tables."""
    try:
        application.state.bundle = load_bundle(ARTIFACT)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(f"{error} Run `uv run python -m src.train` first.") from error
    try:
        database.init_db()
    except (database.DatabaseError, ValueError) as error:
        raise RuntimeError(
            f"Could not initialize Supabase Postgres at {database.server_label()} "
            f"(schema {database.DB_SCHEMA!r}): {error}. "
            "Check SUPABASE_DB_URL in .env."
        ) from error
    LOGGER.info(
        "supabase ready: %s schema=%s",
        database.server_label(),
        database.DB_SCHEMA,
    )
    yield


app = FastAPI(
    title="Heart disease prediction API",
    version="0.1.0",
    description="Predict the likelihood of heart disease from 13 clinical measurements.",
    lifespan=lifespan,
)


def _model_version() -> str | None:
    metrics = app.state.bundle.get("metrics", {})
    version = metrics.get("model_version")
    return str(version) if version is not None else None


def _identity(patient: Patient) -> dict[str, str | None]:
    """Trim identity fields and treat empty strings as 'not provided'."""
    cleaned: dict[str, str | None] = {}
    for field in IDENTITY_FIELDS:
        value = getattr(patient, field)
        cleaned[field] = (value.strip() or None) if isinstance(value, str) else value
    return cleaned


def _persist(prediction: Prediction, payload: dict, identity: dict[str, str | None]) -> None:
    """Store the scored record; failures surface as HTTP 500, never silent data loss."""
    try:
        saved = database.save_prediction(
            features={feature: payload[feature] for feature in FEATURES},
            prediction=prediction.prediction,
            risk_probability=prediction.risk_probability,
            risk_label=prediction.risk_label,
            model_version=prediction.model_version,
            **identity,
        )
    except database.DatabaseError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction succeeded but could not be saved to the "
                f"{database.DB_SCHEMA!r} Supabase schema: {error}"
            ),
        ) from error
    prediction.record_id = saved["record_id"]
    prediction.patient_id = saved["patient_id"]


def _score(frame: pd.DataFrame) -> list[Prediction]:
    """Run the model pipeline and map failures to HTTP errors."""
    if frame.empty:
        raise HTTPException(status_code=422, detail="Provide at least one row to score.")
    try:
        result = predict(app.state.bundle, frame)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    version = _model_version()
    predictions = [
        Prediction(**row, model_version=version) for row in result.to_dict(orient="records")
    ]
    mean_probability = sum(item.risk_probability for item in predictions) / len(predictions)
    LOGGER.info(
        "scored rows=%d mean_probability=%.3f model_version=%s",
        len(predictions),
        mean_probability,
        version,
    )
    return predictions


@app.get("/", tags=["meta"])
def home() -> dict[str, str]:
    """Service metadata and links."""
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "endpoints": (
            "GET /health, POST /predict, POST /predict-batch, POST /predict-csv, "
            "POST /auth/register, POST /auth/login, POST /auth/logout, GET /auth/me, "
            "patient: GET /me/history, "
            "admin only: GET /patients, GET /patients/{patient_id}/history, GET /predictions"
        ),
        "disclaimer": "Not medical advice. Demo use only.",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Service status plus the metrics recorded at training time."""
    metrics = app.state.bundle.get("metrics", {})
    return {"status": "ok", **metrics}


_bearer = HTTPBearer(auto_error=False, description="Token from POST /auth/login")
BearerToken = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def optional_account(credentials: BearerToken) -> dict | None:
    """The signed-in account, or ``None`` when no token is sent; a bad token is rejected."""
    if credentials is None:
        return None
    account = auth.current_session(credentials.credentials)
    if account is None:
        raise _unauthorized("Your session has expired or is invalid. Please sign in again.")
    return account


OptionalAccount = Annotated[dict | None, Depends(optional_account)]


def require_account(account: OptionalAccount) -> dict:
    if account is None:
        raise _unauthorized("Sign-in required.")
    return account


Account = Annotated[dict, Depends(require_account)]


def require_admin(account: OptionalAccount) -> dict:
    """Allow the request only for a signed-in admin."""
    if account is None:
        raise _unauthorized("Admin sign-in required to view patient records.")
    if account["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admins can view other patients' records.")
    return account



@app.post("/auth/login", response_model=Session, tags=["auth"])
def login(credentials: LoginRequest) -> Session:
    """Exchange a username and password (admin or patient) for a bearer token."""
    session = auth.login(credentials.username, credentials.password)
    if session is None:
        raise _unauthorized("Incorrect username or password.")
    return Session(**session)


@app.post("/auth/register", response_model=Session, status_code=201, tags=["auth"])
def register(details: RegisterRequest) -> Session:
    """Patient self-sign-up: creates a new patient record and signs the patient in."""
    try:
        auth.register_patient(details.username, details.password, details.name)
    except ValueError as error:
        status = 409 if "taken" in str(error) else 422
        raise HTTPException(status_code=status, detail=str(error)) from error
    return Session(**auth.login(details.username, details.password))


@app.post("/auth/logout", tags=["auth"])
def logout(account: Account, credentials: BearerToken) -> dict[str, str]:
    """End the current session."""
    auth.logout(credentials.credentials)
    return {"status": "signed out", "username": account["username"]}


@app.get("/auth/me", tags=["auth"])
def me(account: Account) -> dict:
    """The account the token belongs to."""
    return {key: account[key] for key in ("username", "role", "patient_id", "name")}


@app.get("/me/history", tags=["history"])
def my_history(account: Account) -> dict:
    """A signed-in patient's own record and prediction history, newest first."""
    if account["role"] != "patient":
        raise HTTPException(
            status_code=403, detail="Only patient accounts have a personal history."
        )
    record = database.get_patient_with_history(account["patient_id"])
    if record is None:
        raise HTTPException(status_code=404, detail="Your patient record was not found.")
    return record


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict_single(patient: Patient, account: OptionalAccount) -> Prediction:
    """Score one patient, attach tiered guidance, and store the record in the database.

    A signed-in patient's assessment is always saved to their own record; any
    identity fields in the body are ignored for them.
    """
    payload = patient.model_dump()
    scored = _score(pd.DataFrame([payload]))[0]
    scored.guidance = Guidance(
        **recommendations(float(scored.risk_probability), payload)
    )
    identity = _identity(patient)
    if account and account["role"] == "patient":
        identity = {"patient_id": account["patient_id"], "name": None, "notes": None}
    _persist(scored, payload, identity)
    LOGGER.info(
        "saved record_id=%s patient_id=%s", scored.record_id, scored.patient_id
    )
    return scored


@app.post("/predict-batch", response_model=list[Prediction], tags=["prediction"])
def predict_batch(patients: list[Patient]) -> list[Prediction]:
    """Score a list of patient records (max 1000 rows)."""
    if not patients:
        raise HTTPException(status_code=422, detail="Provide at least one patient record.")
    if len(patients) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large: {len(patients)} rows (limit {MAX_BATCH_ROWS}).",
        )
    return _score(pd.DataFrame([patient.model_dump() for patient in patients]))


@app.post("/predict-csv", tags=["prediction"])
def predict_csv(file: UploadFile) -> Response:
    """Score an uploaded CSV that contains the 13 feature columns; returns CSV."""
    payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {error}") from error
    results = _score(frame)
    output = pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame(
                [
                    item.model_dump(exclude={"guidance", "record_id", "patient_id"})
                    for item in results
                ]
            ),
        ],
        axis=1,
    )
    return Response(content=output.to_csv(index=False), media_type="text/csv")


@app.get("/patients", tags=["history"], dependencies=[Depends(require_admin)])
def list_patients() -> dict:
    """Every stored patient with prediction counts and latest score time."""
    patients = database.list_patients()
    return {"count": len(patients), "patients": patients}


@app.get("/patients/{patient_id}", tags=["history"], dependencies=[Depends(require_admin)])
def read_patient(patient_id: str) -> dict:
    """Profile row for one patient id."""
    patient = database.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"No patient with id {patient_id!r}.")
    return patient


@app.get(
    "/patients/{patient_id}/history", tags=["history"], dependencies=[Depends(require_admin)]
)
def patient_history(patient_id: str) -> dict:
    """Full prediction history for one patient, newest first."""
    record = database.get_patient_with_history(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No patient with id {patient_id!r}.")
    return record


@app.get("/predictions", tags=["history"], dependencies=[Depends(require_admin)])
def recent_predictions(limit: int = Query(50, ge=1, le=200)) -> dict:
    """Most recent predictions across all patients (anonymous rows included)."""
    rows = database.recent_predictions(limit=limit)
    return {"count": len(rows), "predictions": rows}
