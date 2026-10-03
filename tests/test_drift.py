import numpy as np
import pytest

from src.drift import (
    build_reference,
    compute_drift,
    load_reference,
    psi,
    worst_status,
    write_reference,
)


def test_psi_zero_for_identical_distributions():
    counts = np.array([50.0, 50.0])
    assert psi(counts, counts) == pytest.approx(0.0, abs=1e-9)


def test_psi_large_for_swapped_distributions():
    score = psi(np.array([90.0, 10.0]), np.array([10.0, 90.0]))
    assert score > 0.25


def test_identical_batch_reports_no_drift(heart_frame):
    reference = build_reference(heart_frame)

    report = compute_drift(reference, heart_frame)

    assert set(report) == set(reference["features"])
    assert all(entry["psi"] < 0.1 for entry in report.values())
    assert worst_status(report) == "ok"


def test_shifted_feature_is_flagged(heart_frame):
    reference = build_reference(heart_frame)
    shifted = heart_frame.copy()
    shifted["age"] = shifted["age"] + 40

    report = compute_drift(reference, shifted)

    assert report["age"]["status"] == "significant"
    assert worst_status(report) == "significant"


def test_missing_feature_is_reported(heart_frame):
    reference = build_reference(heart_frame)

    report = compute_drift(reference, heart_frame.drop(columns=["age"]))

    assert report["age"]["status"] == "missing"


def test_write_and_load_reference_roundtrip(tmp_path, heart_frame):
    path = tmp_path / "reference.json"

    write_reference(path, heart_frame, data_hash="abc123")
    reference = load_reference(path)

    assert reference["rows"] == len(heart_frame)
    assert reference["data_hash"] == "abc123"
    assert set(reference["features"]) == set(heart_frame.columns) - {"target"}


def test_load_reference_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        load_reference(tmp_path / "nope.json")
