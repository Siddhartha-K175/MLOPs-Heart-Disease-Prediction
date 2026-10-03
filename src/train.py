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
from urllib.request import urlopen

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .drift import write_reference
from .model import FEATURES, range_violations

LOGGER = logging.getLogger(__name__)
DEFAULT_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
TARGET = "target"
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]
CV_FOLDS = 5
SPLIT_NAMES = ("train", "validate", "test")
DEFAULT_SPLITS = {name: Path("data") / f"{name}.csv" for name in SPLIT_NAMES}


def load_data(source: str | Path | None) -> pd.DataFrame:
    """Load a local CSV or the pinned public sample dataset."""
    if source:
        frame = pd.read_csv(source)
    else:
        LOGGER.info("Downloading dataset from %s", DEFAULT_URL)
        with urlopen(DEFAULT_URL, timeout=30) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
            raw_name = "processed.cleveland.data"
            with archive.open(raw_name) as raw_data:
                frame = pd.read_csv(raw_data, header=None, na_values="?")
            frame.columns = FEATURES + [TARGET]
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
    violations = range_violations(frame)
    if violations:
        details = "; ".join(f"{name}: {message}" for name, message in violations.items())
        raise ValueError(f"Dataset failed range checks: {details}")
    if frame[TARGET].nunique() < 2:
        raise ValueError("The target column must contain both classes.")
    return frame


def data_hash(*frames: pd.DataFrame) -> str:
    """Stable fingerprint of the rows used for a training run."""
    payload = pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_pipeline() -> Pipeline:
    """Build a preprocessing and logistic-regression pipeline."""
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
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def evaluate(pipeline: Pipeline, frame: pd.DataFrame) -> tuple[float, float]:
    """Return (accuracy, roc_auc) for a fitted pipeline on a labelled frame."""
    features = frame[FEATURES]
    target = frame[TARGET]
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return (
        round(float(accuracy_score(target, predictions)), 4),
        round(float(roc_auc_score(target, probabilities)), 4),
    )


def _cross_validate(fit_frame: pd.DataFrame) -> dict[str, float]:
    """5-fold stratified cross-validation over the model-selection data."""
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = cross_validate(
        build_pipeline(),
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
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "heart-disease",
    registered_model_name: str = "heart-disease-classifier",
    log_to_mlflow: bool = True,
) -> dict[str, float | int | str]:
    """Train, evaluate, and persist the model and metrics.

    With `validation`/`test` frames the run uses 5-fold CV on train+validate,
    refits on that combined data, and reports the untouched test frame once.
    Without them the legacy 80/20 random split applies.
    """
    split_frames = [data]
    if validation is not None:
        split_frames.append(validation)
    if test is not None:
        split_frames.append(test)
    metrics: dict[str, float | int | str] = {
        "model": "logistic_regression",
        "features": FEATURES,
        "data_hash": data_hash(*split_frames),
    }

    if validation is None and test is None:
        x_train, x_test, y_train, y_test = train_test_split(
            data[FEATURES], data[TARGET], test_size=0.2, random_state=42, stratify=data[TARGET]
        )
        pipeline = build_pipeline()
        pipeline.fit(x_train, y_train)
        accuracy, roc_auc = evaluate(pipeline, pd.concat([x_test, y_test], axis=1))
        metrics |= {
            "dataset_rows": len(data),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "accuracy": accuracy,
            "roc_auc": roc_auc,
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
            **_cross_validate(fit_frame),
        }
        pipeline = build_pipeline()
        pipeline.fit(fit_frame[FEATURES], fit_frame[TARGET])
        holdout = test if test is not None else validation
        assert holdout is not None
        accuracy, roc_auc = evaluate(pipeline, holdout)
        metrics |= {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "holdout": "test" if test is not None else "validation",
        }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_reference(artifact_dir / "reference.json", fit_frame, str(metrics["data_hash"]))

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
                    "holdout": metrics["holdout"],
                    "data_hash": metrics["data_hash"],
                    "features": ",".join(FEATURES),
                }
            )
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, int | float) and not isinstance(value, bool)
                }
            )
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
    return parser


def _resolve_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Decide between explicit splits, default splits, a single CSV, or download."""
    explicit = [getattr(args, name) for name in SPLIT_NAMES]
    if args.data and any(explicit):
        raise ValueError("--data cannot be combined with --train/--validate/--test.")
    if args.data:
        return load_data(args.data), None, None
    if all(explicit):
        return load_data(explicit[0]), load_data(explicit[1]), load_data(explicit[2])
    if any(explicit):
        raise ValueError("--train, --validate and --test must be provided together.")
    defaults = list(DEFAULT_SPLITS.values())
    if all(path.exists() for path in defaults):
        LOGGER.info("Using splits from data/train.csv, data/validate.csv, data/test.csv")
        return (
            load_data(DEFAULT_SPLITS["train"]),
            load_data(DEFAULT_SPLITS["validate"]),
            load_data(DEFAULT_SPLITS["test"]),
        )
    LOGGER.info("No local splits found; downloading the public dataset instead")
    return load_data(None), None, None


def main() -> int:
    """Run training and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = create_parser().parse_args()
    try:
        train_frame, validation_frame, test_frame = _resolve_inputs(args)
        metrics = train(
            train_frame,
            args.artifacts,
            validation=validation_frame,
            test=test_frame,
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
