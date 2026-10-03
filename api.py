"""FastAPI service exposing the heart-disease model.

Usage:
    uv run uvicorn api:app --reload --port 8000
    Open http://127.0.0.1:8000/docs for the interactive API reference.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, create_model

from src.model import FEATURE_BOUNDS, FEATURE_LABELS, FEATURES, load_bundle, predict

LOGGER = logging.getLogger("heart.api")
ROOT = Path(__file__).parent
ARTIFACT = ROOT / "artifacts" / "heart_model.joblib"
MAX_BATCH_ROWS = 1000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

DEFAULTS = {
    "age": 55.0, "sex": 1, "cp": 1, "trestbps": 130.0, "chol": 240.0,
    "fbs": 0, "restecg": 1, "thalach": 150.0, "exang": 0, "oldpeak": 1.0,
    "slope": 1, "ca": 0, "thal": 2,
}

Patient = create_model(
    "Patient",
    **{
        feature: (
            float,
            Field(
                float(DEFAULTS[feature]),
                ge=FEATURE_BOUNDS[feature][0],
                le=FEATURE_BOUNDS[feature][1],
                description=FEATURE_LABELS[feature],
            ),
        )
        for feature in FEATURES
    },
)


class Prediction(BaseModel):
    prediction: int
    risk_probability: float
    risk_label: str
    model_version: str | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the trained model bundle before serving traffic."""
    try:
        application.state.bundle = load_bundle(ARTIFACT)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(f"{error} Run `uv run python -m src.train` first.") from error
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
        "endpoints": "GET /health, POST /predict, POST /predict-batch, POST /predict-csv",
        "disclaimer": "Not medical advice. Demo use only.",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Service status plus the metrics recorded at training time."""
    metrics = app.state.bundle.get("metrics", {})
    return {"status": "ok", **metrics}


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict_single(patient: Patient) -> Prediction:
    """Score one patient record."""
    return _score(pd.DataFrame([patient.model_dump()]))[0]


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
        [frame.reset_index(drop=True), pd.DataFrame([item.model_dump() for item in results])],
        axis=1,
    )
    return Response(content=output.to_csv(index=False), media_type="text/csv")
