"""Population-stability (PSI) drift checks against the training reference.

Usage:
    python -m src.drift --data data/new_batch.csv
    python -m src.drift --data data/new_batch.csv --reference artifacts/reference.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .model import FEATURES

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6
MODERATE_PSI = 0.1
SIGNIFICANT_PSI = 0.25
MIN_BATCH_ROWS = 100
DEFAULT_REFERENCE = Path("artifacts/reference.json")


def build_reference(frame: pd.DataFrame, data_hash: str = "") -> dict:
    """Summarise the training distribution per feature for later PSI comparison."""
    features: dict[str, dict] = {}
    for feature in FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").astype(float)
        if values.nunique() <= 10:
            counts = values.value_counts()
            features[feature] = {
                "type": "categorical",
                "values": {str(key): int(count) for key, count in counts.items()},
            }
        else:
            edges = np.quantile(values.dropna(), np.linspace(0, 1, 11))
            counts, _ = np.histogram(values.dropna(), bins=edges)
            features[feature] = {
                "type": "numeric",
                "edges": [round(float(edge), 6) for edge in edges],
                "counts": [int(count) for count in counts],
            }
    return {"rows": int(len(frame)), "data_hash": data_hash, "features": features}


def write_reference(path: Path, frame: pd.DataFrame, data_hash: str = "") -> None:
    """Persist the training reference used by the drift CLI."""
    path.write_text(json.dumps(build_reference(frame, data_hash), indent=2), encoding="utf-8")


def load_reference(path: Path) -> dict:
    """Load a reference file written by write_reference."""
    if not path.exists():
        raise FileNotFoundError(f"Reference not found at {path}. Train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _proportions(counts: np.ndarray) -> np.ndarray:
    smoothed = np.asarray(counts, dtype=float) + EPSILON
    return smoothed / smoothed.sum()


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index between two binned distributions."""
    expected_p = _proportions(expected)
    actual_p = _proportions(actual)
    return float(np.sum((actual_p - expected_p) * np.log(actual_p / expected_p)))


def _status(score: float) -> str:
    if score >= SIGNIFICANT_PSI:
        return "significant"
    if score >= MODERATE_PSI:
        return "moderate"
    return "ok"


def compute_drift(reference: dict, frame: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    """Compare a batch against the reference; returns {feature: {psi, status}}."""
    report: dict[str, dict[str, float | str]] = {}
    for feature, spec in reference["features"].items():
        if feature not in frame.columns:
            report[feature] = {"psi": float("nan"), "status": "missing"}
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").dropna().astype(float)
        if spec["type"] == "numeric":
            edges = np.asarray(spec["edges"], dtype=float)
            expected = np.asarray(spec["counts"], dtype=float)
            actual, _ = np.histogram(values.to_numpy(), bins=edges)
            score = psi(expected, actual)
        else:
            expected = np.asarray(list(spec["values"].values()), dtype=float)
            keys = list(spec["values"].keys())
            actual = np.asarray(
                [float((values == float(key)).sum()) for key in keys], dtype=float
            )
            outside = float(len(values) - actual.sum())
            score = psi(np.append(expected, 0.0), np.append(actual, outside))
        report[feature] = {"psi": round(score, 4), "status": _status(score)}
    return report


def worst_status(report: dict[str, dict[str, float | str]]) -> str:
    """Return the most severe status present in a drift report."""
    for level in ("significant", "moderate", "ok"):
        if any(entry["status"] == level for entry in report.values()):
            return level
    return "ok"


def create_parser() -> argparse.ArgumentParser:
    """Create the drift CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="CSV batch to compare.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 when any feature drifts significantly.",
    )
    return parser


def main() -> int:
    """Run a drift report and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = create_parser().parse_args()
    try:
        reference = load_reference(args.reference)
        batch = pd.read_csv(args.data)
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    report = compute_drift(reference, batch)
    for feature in FEATURES:
        entry = report.get(feature)
        if entry:
            LOGGER.info("%-10s psi=%-8s status=%s", feature, entry["psi"], entry["status"])
    if len(batch) < MIN_BATCH_ROWS:
        LOGGER.warning(
            "Batch has %d rows; PSI is unreliable below %d rows.", len(batch), MIN_BATCH_ROWS
        )
    overall = worst_status(report)
    LOGGER.info("Overall drift: %s (%d rows compared)", overall, len(batch))
    if args.fail_on_drift and overall == "significant":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
