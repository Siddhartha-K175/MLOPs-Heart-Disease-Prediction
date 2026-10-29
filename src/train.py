"""Train and evaluate the heart-disease model.

Usage:
    python -m src.train
    python -m src.train --data data/heart.csv
    python -m src.train --train data/train.csv --validate data/validate.csv --test data/test.csv
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .drift import write_reference
from .model import FEATURES, range_violations
from .profile import combine_profiles, data_profile, summarize_profile

LOGGER = logging.getLogger(__name__)
DEFAULT_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
TARGET = "target"
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]
CV_FOLDS = 5
SPLIT_NAMES = ("train", "validate", "test")
DEFAULT_SPLITS = {name: Path("data") / f"{name}.csv" for name in SPLIT_NAMES}


def fetch_raw(source: str | Path | None) -> pd.DataFrame:
    """Read a local CSV or download the pinned public dataset, uncleaned."""
    if source:
        return pd.read_csv(source)
    LOGGER.info("Downloading dataset from %s", DEFAULT_URL)
    with urlopen(DEFAULT_URL, timeout=30) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        raw_name = "processed.cleveland.data"
        with archive.open(raw_name) as raw_data:
            frame = pd.read_csv(raw_data, header=None, na_values="?")
    frame.columns = FEATURES + [TARGET]
    return frame


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a raw frame so it matches the training schema."""
    frame.columns = [column.strip().lower() for column in frame.columns]
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    frame = frame[FEATURES + [TARGET]].copy()
    for column in FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    incomplete = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna()
    if incomplete:
        LOGGER.warning("Dropped %d row(s) with missing values", incomplete)
    frame[TARGET] = (frame[TARGET] > 0).astype(int)
    before = len(frame)
    frame = frame.drop_duplicates()
    removed = before - len(frame)
    if removed:
        LOGGER.warning("Dropped %d duplicate row(s)", removed)
    violations = range_violations(frame)
    if violations:
        details = "; ".join(f"{name}: {message}" for name, message in violations.items())
        raise ValueError(f"Dataset failed range checks: {details}")
    if frame[TARGET].nunique() < 2:
        raise ValueError("The target column must contain both classes.")
    return frame


def load_data(source: str | Path | None) -> pd.DataFrame:
    """Load a local CSV or the pinned public sample dataset."""
    return validate_frame(fetch_raw(source))


def load_profiled(source: str | Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a source and return (cleaned frame, pre-cleaning data profile)."""
    raw = fetch_raw(source)
    profile = data_profile(raw)
    return validate_frame(raw.copy()), profile


def data_hash(*frames: pd.DataFrame) -> str:
    """Stable fingerprint of the rows used for a training run."""
    payload = pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_pipeline(*, smote: bool = False) -> Pipeline | ImbPipeline:
    """Build a preprocessing and logistic-regression pipeline.

    With `smote=True`, an SMOTE step runs after scaling so each cross-validation
    fold resamples only its own training portion (no leakage into validation rows).
    """
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    steps = [("preprocessor", preprocessor)]
    if smote:
        steps.append(("sampler", SMOTE(random_state=42)))
    steps.append(("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")))
    pipeline = ImbPipeline if smote else Pipeline
    return pipeline(steps=steps)


def evaluate(pipeline: Pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    """Return holdout metrics, including the confusion matrix, for a fitted pipeline."""
    features = frame[FEATURES]
    target = frame[TARGET]
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        target, predictions, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": round(float(accuracy_score(target, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
        "precision": round(float(precision_score(target, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(target, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(target, predictions, zero_division=0)), 4),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "confusion_matrix": [
            [int(true_negatives), int(false_positives)],
            [int(false_negatives), int(true_positives)],
        ],
    }


def feature_importance(pipeline: Pipeline | ImbPipeline) -> pd.DataFrame:
    """Coefficient table for the fitted logistic model, largest effect first."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": [str(name).split("__", 1)[-1] for name in names],
            "coefficient": coefficients,
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    table["odds_ratio"] = np.exp(table["coefficient"])
    table = table.sort_values("abs_coefficient", ascending=False, ignore_index=True)
    for column in ("coefficient", "abs_coefficient", "odds_ratio"):
        table[column] = table[column].astype(float).round(4)
    return table


def _cross_validate(fit_frame: pd.DataFrame, *, smote: bool = False) -> dict[str, float]:
    """5-fold stratified cross-validation over the model-selection data."""
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = cross_validate(
        build_pipeline(smote=smote),
        fit_frame[FEATURES],
        fit_frame[TARGET],
        cv=splitter,
        scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
    )
    return {
        "cv_accuracy": round(float(np.mean(results["test_accuracy"])), 4),
        "cv_accuracy_std": round(float(np.std(results["test_accuracy"])), 4),
        "cv_roc_auc": round(float(np.mean(results["test_roc_auc"])), 4),
        "cv_roc_auc_std": round(float(np.std(results["test_roc_auc"])), 4),
    }


def _register_champion(name: str, model_info: object) -> str | None:
    """Point the `champion` alias at the freshly registered version."""
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        return None
    try:
        MlflowClient().set_registered_model_alias(name, "champion", int(version))
    except (MlflowException, AttributeError) as error:
        LOGGER.warning("Could not set champion alias: %s", error)
    return str(version)


def train(
    data: pd.DataFrame,
    artifact_dir: Path,
    *,
    validation: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    profile: dict[str, Any] | None = None,
    smote: bool = False,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "heart-disease",
    registered_model_name: str = "heart-disease-classifier",
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, and persist the model and metrics.

    With `validation`/`test` frames the run uses 5-fold CV on train+validate,
    refits on that combined data, and reports the untouched test frame once.
    Without them the legacy 80/20 random split applies.

    `profile` holds pre-cleaning missingness/duplicate counts; `smote=True`
    resamples each training fold with SMOTE.
    """
    split_frames = [data]
    if validation is not None:
        split_frames.append(validation)
    if test is not None:
        split_frames.append(test)
    metrics: dict[str, Any] = {
        "model": "logistic_regression",
        "features": FEATURES,
        "data_hash": data_hash(*split_frames),
        "smote": bool(smote),
        "profile": summarize_profile(profile if profile is not None else data_profile(data)),
    }

    if validation is None and test is None:
        x_train, x_test, y_train, y_test = train_test_split(
            data[FEATURES], data[TARGET], test_size=0.2, random_state=42, stratify=data[TARGET]
        )
        pipeline = build_pipeline(smote=smote)
        pipeline.fit(x_train, y_train)
        metrics |= {
            "dataset_rows": len(data),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            **evaluate(pipeline, pd.concat([x_test, y_test], axis=1)),
            "holdout": "random_split",
        }
        fit_frame = data
    else:
        selection_frames = [data] if validation is None else [data, validation]
        fit_frame = pd.concat(selection_frames, ignore_index=True)
        metrics |= {
            "dataset_rows": len(fit_frame),
            "train_rows": len(data),
            "validation_rows": len(validation) if validation is not None else 0,
            "test_rows": len(test) if test is not None else 0,
            "cv_folds": CV_FOLDS,
            **_cross_validate(fit_frame, smote=smote),
        }
        pipeline = build_pipeline(smote=smote)
        pipeline.fit(fit_frame[FEATURES], fit_frame[TARGET])
        holdout = test if test is not None else validation
        assert holdout is not None
        metrics |= {
            **evaluate(pipeline, holdout),
            "holdout": "test" if test is not None else "validation",
        }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_reference(artifact_dir / "reference.json", fit_frame, str(metrics["data_hash"]))
    importance = feature_importance(pipeline)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    metrics["top_feature"] = str(importance.loc[0, "feature"])

    def write_metrics_file() -> None:
        (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if log_to_mlflow:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            metrics["mlflow_run_id"] = run.info.run_id
            mlflow.log_params(
                {
                    "model": "logistic_regression",
                    "class_weight": "balanced",
                    "cv_folds": CV_FOLDS,
                    "random_state": 42,
                    "smote": str(smote),
                    "holdout": metrics["holdout"],
                    "top_feature": metrics["top_feature"],
                    "data_hash": metrics["data_hash"],
                    "features": ",".join(FEATURES),
                }
            )
            logged = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
            logged |= {
                f"profile_{key}": float(metrics["profile"][key])
                for key in ("rows", "missing_cells_pct", "duplicate_rows", "rows_with_missing_pct")
            }
            mlflow.log_metrics(logged)
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=registered_model_name,
            )
            metrics["registered_model"] = registered_model_name
            version = _register_champion(registered_model_name, model_info)
            if version is not None:
                metrics["model_version"] = version
            write_metrics_file()
            mlflow.log_artifact(str(artifact_dir / "metrics.json"))
            mlflow.log_artifact(str(artifact_dir / "reference.json"))
            mlflow.log_artifact(str(artifact_dir / "feature_importance.csv"))
    else:
        write_metrics_file()

    import joblib

    joblib.dump({"model": pipeline, "metrics": metrics}, artifact_dir / "heart_model.joblib")
    return metrics


def create_parser() -> argparse.ArgumentParser:
    """Create the training CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, help="Optional CSV path with the required heart columns."
    )
    parser.add_argument("--train", type=Path, help="Training split CSV.")
    parser.add_argument("--validate", type=Path, help="Validation split CSV.")
    parser.add_argument("--test", type=Path, help="Holdout test split CSV.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="heart-disease")
    parser.add_argument("--registered-model", default="heart-disease-classifier")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow tracking and only write local artifacts.",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Resample training folds with SMOTE to balance the classes.",
    )
    return parser


def _load_many(paths: list[Path]) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Load several sources, returning their frames and a combined profile."""
    frames: list[pd.DataFrame] = []
    profiles: list[dict[str, Any]] = []
    for path in paths:
        frame, profile = load_profiled(path)
        frames.append(frame)
        profiles.append(profile)
    return frames, combine_profiles(profiles)


def _resolve_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
    """Decide between explicit splits, default splits, a single CSV, or download.

    Returns (train, validation, test, combined pre-cleaning profile).
    """
    explicit = [getattr(args, name) for name in SPLIT_NAMES]
    if args.data and any(explicit):
        raise ValueError("--data cannot be combined with --train/--validate/--test.")
    if args.data:
        frame, profile = load_profiled(args.data)
        return frame, None, None, profile
    if all(explicit):
        frames, profile = _load_many(explicit)
        return frames[0], frames[1], frames[2], profile
    if any(explicit):
        raise ValueError("--train, --validate and --test must be provided together.")
    defaults = list(DEFAULT_SPLITS.values())
    if all(path.exists() for path in defaults):
        LOGGER.info("Using splits from data/train.csv, data/validate.csv, data/test.csv")
        frames, profile = _load_many(defaults)
        return frames[0], frames[1], frames[2], profile
    LOGGER.info("No local splits found; downloading the public dataset instead")
    frame, profile = load_profiled(None)
    return frame, None, None, profile


def main() -> int:
    """Run training and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = create_parser().parse_args()
    try:
        train_frame, validation_frame, test_frame, profile = _resolve_inputs(args)
        metrics = train(
            train_frame,
            args.artifacts,
            validation=validation_frame,
            test=test_frame,
            profile=profile,
            smote=args.smote,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.experiment,
            registered_model_name=args.registered_model,
            log_to_mlflow=not args.no_mlflow,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    LOGGER.info("Model saved. Metrics: %s", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
