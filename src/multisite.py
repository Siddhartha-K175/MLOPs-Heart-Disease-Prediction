"""Multi-hospital UCI heart-disease data and external (cross-hospital) validation.

The UCI Heart Disease collection comes from four hospitals: Cleveland (the data
the production model is trained on), Hungary, Switzerland and the VA Long Beach.
External validation trains on some hospitals and tests on one the model has
never seen, which shows whether it generalises beyond a single clinic:

* ``cleveland_to_other_sites``: train on Cleveland, as the production model does,
  and test on each other hospital.
* ``leave_one_site_out``: for each hospital, train on the other three and test on it.

Both run for two feature sets: all 13 features, and a "core" set without ca,
thal and slope, which most non-Cleveland patients never had measured. Missing
values are imputed (median / most frequent). No missing-value indicator columns
are added on purpose: which values are missing mostly reveals the hospital, and
hospitals have very different disease rates, so indicators would teach the model
the hospital rather than the disease.

Usage:
    python -m src.multisite                 # writes artifacts/site_validation.json
    python -m src.multisite --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .model import FEATURE_BOUNDS, FEATURES
from .train import CORE_FEATURES, CV_FOLDS, DEFAULT_URL, TARGET, build_pipeline, evaluate

LOGGER = logging.getLogger(__name__)
SITE = "site"
SITES = {
    "cleveland": "Cleveland Clinic Foundation (US)",
    "hungarian": "Hungarian Institute of Cardiology, Budapest",
    "switzerland": "University Hospitals of Zurich and Basel",
    "va": "VA Medical Center, Long Beach (US)",
}
DEVELOPMENT_SITE = "cleveland"
ARCHIVE = Path("data/uci/heart+disease.zip")
# Recorded as 0 where the measurement was not taken (e.g. all of Switzerland's cholesterol).
ZERO_MEANS_MISSING = ("trestbps", "chol")
FEATURE_SETS = {"all_features": FEATURES, "core_features": CORE_FEATURES}
REPORTED = ("roc_auc", "accuracy", "recall", "precision")


def fetch_archive(path: Path = ARCHIVE) -> Path:
    """Download the UCI Heart Disease archive once and reuse the local copy."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading %s", DEFAULT_URL)
        with urlopen(DEFAULT_URL, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def clean_site(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce one hospital's raw rows into the training schema, keeping gaps as NaN.

    Placeholder zeros and values outside FEATURE_BOUNDS become missing (to be
    imputed) instead of failing the whole file. Returns (frame, counts of values
    blanked per feature).
    """
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(subset=[TARGET])
    blanked: dict[str, int] = {}
    for feature in FEATURES:
        low, high = FEATURE_BOUNDS[feature]
        values = frame[feature]
        invalid = values.notna() & ((values < low) | (values > high))
        if feature in ZERO_MEANS_MISSING:
            invalid |= values == 0
        if invalid.any():
            blanked[feature] = int(invalid.sum())
            frame.loc[invalid, feature] = np.nan
    frame[TARGET] = (frame[TARGET] > 0).astype(int)
    return frame.drop_duplicates().reset_index(drop=True), blanked


def load_sites(archive: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All four hospitals in one frame with a ``site`` column, plus a per-site summary."""
    frames = []
    summary: dict[str, Any] = {}
    with zipfile.ZipFile(fetch_archive(archive or ARCHIVE)) as bundle:
        for site, hospital in SITES.items():
            with bundle.open(f"processed.{site}.data") as raw:
                frame = pd.read_csv(raw, header=None, na_values="?", names=FEATURES + [TARGET])
            frame, blanked = clean_site(frame)
            frame[SITE] = site
            frames.append(frame)
            missing = frame[FEATURES].isna().mean()
            summary[site] = {
                "hospital": hospital,
                "rows": len(frame),
                "disease_rate": round(float(frame[TARGET].mean()), 4),
                "invalid_values_blanked": blanked,
                "missing_pct": {
                    feature: round(float(share) * 100, 1)
                    for feature, share in missing.items()
                    if share > 0
                },
            }
    return pd.concat(frames, ignore_index=True), summary


def score(pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    """Discrimination, threshold and calibration metrics on one hospital's rows."""
    metrics = evaluate(pipeline, frame)
    probabilities = pipeline.predict_proba(frame[FEATURES])[:, 1]
    return {
        "rows": len(frame),
        "disease_rate": round(float(frame[TARGET].mean()), 4),
        **{key: metrics[key] for key in REPORTED},
        "mean_predicted_risk": round(float(probabilities.mean()), 4),
        "brier": round(float(brier_score_loss(frame[TARGET], probabilities)), 4),
    }


def site_validation(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Internal CV on Cleveland, Cleveland -> each other site, and leave-one-site-out."""
    development = frame[frame[SITE] == DEVELOPMENT_SITE]
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    internal = cross_val_score(
        build_pipeline(features=features),
        development[FEATURES],
        development[TARGET],
        cv=splitter,
        scoring="roc_auc",
    )
    model = build_pipeline(features=features).fit(development[FEATURES], development[TARGET])
    from_cleveland = {
        site: score(model, frame[frame[SITE] == site])
        for site in SITES
        if site != DEVELOPMENT_SITE
    }
    leave_one_out = {}
    for site in SITES:
        held_out = frame[SITE] == site
        model = build_pipeline(features=features).fit(
            frame.loc[~held_out, FEATURES], frame.loc[~held_out, TARGET]
        )
        leave_one_out[site] = score(model, frame[held_out])
    return {
        "features": list(features),
        "cleveland_internal_cv_roc_auc": round(float(internal.mean()), 4),
        "cleveland_to_other_sites": from_cleveland,
        "leave_one_site_out": leave_one_out,
    }


def run(archive: Path | None = None) -> dict[str, Any]:
    frame, summary = load_sites(archive)
    return {
        "sites": summary,
        "feature_sets": {
            name: site_validation(frame, features) for name, features in FEATURE_SETS.items()
        },
    }


def _flat_metrics(report: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for name, result in report["feature_sets"].items():
        flat[f"{name}.cleveland_internal_cv_roc_auc"] = result["cleveland_internal_cv_roc_auc"]
        for scheme in ("cleveland_to_other_sites", "leave_one_site_out"):
            for site, metrics in result[scheme].items():
                for key in (*REPORTED, "brier"):
                    flat[f"{name}.{scheme}.{site}.{key}"] = float(metrics[key])
    return flat


def _print_table(report: dict[str, Any]) -> None:
    for name, result in report["feature_sets"].items():
        print(f"\n{name} ({len(result['features'])} features) - "
              f"Cleveland internal CV ROC-AUC {result['cleveland_internal_cv_roc_auc']:.3f}")
        for scheme in ("cleveland_to_other_sites", "leave_one_site_out"):
            print(f"  {scheme}:")
            for site, m in result[scheme].items():
                print(f"    {site:12s} n={m['rows']:3d} disease={m['disease_rate']:.0%}  "
                      f"ROC-AUC={m['roc_auc']:.3f}  acc={m['accuracy']:.3f}  "
                      f"recall={m['recall']:.3f}  brier={m['brier']:.3f}  "
                      f"mean risk={m['mean_predicted_risk']:.0%}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Cross-hospital validation of the model.")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=Path("artifacts/site_validation.json"))
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="heart-disease-site-validation")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.archive)
    except (OSError, KeyError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(report)
    if not args.no_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name="site-validation"):
            mlflow.log_params({"development_site": DEVELOPMENT_SITE, "sites": ",".join(SITES)})
            mlflow.log_metrics(_flat_metrics(report))
            mlflow.log_artifact(str(args.output))
    LOGGER.info("Report written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
