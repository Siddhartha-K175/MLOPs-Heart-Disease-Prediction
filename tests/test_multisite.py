"""Tests for multi-hospital loading and cross-site validation (src/multisite.py)."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import multisite
from src.model import FEATURES
from src.train import CORE_FEATURES, TARGET, build_pipeline, train


@pytest.fixture
def archive(tmp_path, heart_frame):
    """A stand-in for the UCI archive: four hospitals of synthetic rows with real quirks."""
    path = tmp_path / "heart+disease.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for offset, site in enumerate(multisite.SITES):
            frame = heart_frame.sample(frac=1.0, random_state=offset).reset_index(drop=True)
            frame[TARGET] = frame[TARGET] * (1 + offset % 3)  # UCI codes disease as 1-4
            text = frame.astype(object)
            if site != "cleveland":
                text.loc[: len(text) // 2, "ca"] = "?"  # mostly unmeasured elsewhere
                text.loc[:5, "chol"] = 0  # placeholder zeros
            bundle.writestr(f"processed.{site}.data", text.to_csv(header=False, index=False))
    return path


def test_clean_site_blanks_placeholders_and_out_of_range_values():
    raw = pd.DataFrame(
        [
            [55, 1, 3, 0, 0, 0, 0, 150, 0, -0.5, 2, 0, 3, 2],
            [60, 1, 4, 140, 250, 1, 2, 120, 1, 2.0, 2, 1, 7, 0],
            [60, 1, 4, 140, 250, 1, 2, 120, 1, 2.0, 2, 1, 7, 0],  # duplicate
        ],
        columns=FEATURES + [TARGET],
    )
    frame, blanked = multisite.clean_site(raw)
    assert len(frame) == 2
    assert frame.loc[0, ["trestbps", "chol", "oldpeak"]].isna().all()
    assert blanked == {"trestbps": 1, "chol": 1, "oldpeak": 1}
    assert frame[TARGET].tolist() == [1, 0]  # 1-4 collapse to "disease"


def test_load_sites_combines_all_hospitals(archive):
    frame, summary = multisite.load_sites(archive)
    assert set(frame[multisite.SITE]) == set(multisite.SITES)
    assert set(summary) == set(multisite.SITES)
    assert summary["hungarian"]["missing_pct"]["ca"] > 40
    assert summary["va"]["invalid_values_blanked"]["chol"] == 6
    assert set(frame[TARGET].unique()) == {0, 1}


def test_pipeline_imputes_and_respects_feature_subset(heart_frame):
    patchy = heart_frame.copy()
    patchy.loc[::3, "chol"] = np.nan
    patchy["ca"] = np.nan  # never measured: ignored by the core model
    model = build_pipeline(features=CORE_FEATURES).fit(patchy[FEATURES], patchy[TARGET])
    names = model.named_steps["preprocessor"].get_feature_names_out()
    columns = [name.split("__", 1)[1] for name in names]
    assert not any(column.startswith(("ca_", "thal_", "slope_")) for column in columns)
    assert "thalach" in columns  # peak heart rate stays
    probabilities = model.predict_proba(patchy[FEATURES])[:, 1]
    assert np.isfinite(probabilities).all()


def test_site_validation_reports_every_scheme(archive):
    frame, _ = multisite.load_sites(archive)
    result = multisite.site_validation(frame, FEATURES)
    assert set(result["cleveland_to_other_sites"]) == {"hungarian", "switzerland", "va"}
    assert set(result["leave_one_site_out"]) == set(multisite.SITES)
    for metrics in result["leave_one_site_out"].values():
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0
        assert metrics["rows"] > 0


def test_cli_writes_report_for_both_feature_sets(archive, tmp_path):
    output = tmp_path / "site_validation.json"
    assert multisite.main(["--archive", str(archive), "--output", str(output), "--no-mlflow"]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["feature_sets"]) == {"all_features", "core_features"}
    assert report["feature_sets"]["core_features"]["features"] == CORE_FEATURES


def test_train_records_dataset_and_feature_subset(tmp_path, heart_frame):
    metrics = train(
        heart_frame, tmp_path, features=CORE_FEATURES, dataset="uci_four_sites",
        log_to_mlflow=False,
    )
    assert metrics["features"] == CORE_FEATURES
    assert metrics["dataset"] == "uci_four_sites"
