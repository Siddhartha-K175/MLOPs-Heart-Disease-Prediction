"""Shared model schema and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest-pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG (0-2)",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression",
    "slope": "ST slope (0-2)",
    "ca": "Major vessels (0-3)",
    "thal": "Thalassemia (0-3)",
}

# Clinically plausible ranges; rows outside them are rejected rather than scored.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 4),
    "trestbps": (50, 300),
    "chol": (100, 600),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 220),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 3),
    "ca": (0, 4),
    "thal": (0, 7),
}


def range_violations(data: pd.DataFrame) -> dict[str, str]:
    """Return {column: message} for every feature value outside FEATURE_BOUNDS."""
    found: dict[str, str] = {}
    for feature in FEATURES:
        if feature not in data.columns:
            continue
        low, high = FEATURE_BOUNDS[feature]
        values = pd.to_numeric(data[feature], errors="coerce")
        out = values.notna() & ((values < low) | (values > high))
        if out.any():
            found[feature] = (
                f"{int(out.sum())} row(s) outside [{low}, {high}] "
                f"(min={values.min()}, max={values.max()})"
            )
    return found



def validate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and order tabular inputs for inference."""
    missing = [feature for feature in FEATURES if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    result = data[FEATURES].copy()
    for feature in FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
    if result.isna().any().any():
        invalid = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Inputs must be numeric and non-empty: {', '.join(invalid)}")
    return result


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a trained model bundle from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m src.train` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Model artifact is invalid: expected a bundle with a `model` key.")
    return bundle


def predict(bundle: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    """Return predictions and probabilities for validated tabular input."""
    features = validate_features(data)
    model = bundle["model"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "prediction": predictions.astype(int),
            "risk_probability": probabilities,
            "risk_label": [
                "Higher likelihood of heart disease"
                if value
                else "Lower likelihood of heart disease"
                for value in predictions
            ],
        },
        index=data.index,
    )
