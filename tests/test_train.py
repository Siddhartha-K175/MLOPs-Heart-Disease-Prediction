import json

import pandas as pd
import pytest

from src.model import FEATURES, load_bundle, predict
from src.train import _resolve_inputs, create_parser, data_hash, load_data, train


def test_load_data_reports_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [50]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_data(path)


def test_load_data_binarizes_target_and_drops_missing(tmp_path, heart_frame):
    frame = heart_frame.copy()
    frame["target"] = frame["target"].map({0: 0, 1: 4})
    frame.loc[frame.index[0], "chol"] = None
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)

    result = load_data(path)

    assert len(result) == len(frame) - 1
    assert set(result["target"].unique()) == {0, 1}


def test_load_data_rejects_out_of_range_values(tmp_path, heart_frame):
    frame = heart_frame.copy()
    frame.loc[frame.index[0], "age"] = 250
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="range checks"):
        load_data(path)


def test_load_data_rejects_single_class_target(tmp_path, heart_frame):
    frame = heart_frame.copy()
    frame["target"] = 1
    path = tmp_path / "single_class.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="both classes"):
        load_data(path)


def test_data_hash_changes_with_rows(heart_frame):
    assert data_hash(heart_frame) == data_hash(heart_frame.copy())
    assert data_hash(heart_frame) != data_hash(heart_frame.head(10))


def test_train_without_mlflow_writes_artifacts(tmp_path, heart_frame):
    artifact_dir = tmp_path / "artifacts"

    metrics = train(heart_frame, artifact_dir, log_to_mlflow=False)

    assert (artifact_dir / "heart_model.joblib").exists()
    assert (artifact_dir / "reference.json").exists()
    saved = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["holdout"] == "random_split"
    assert saved["data_hash"] == metrics["data_hash"]
    assert 0 <= saved["accuracy"] <= 1
    assert "mlflow_run_id" not in saved


def test_train_with_splits_uses_cv_and_holdout(tmp_path, heart_frame):
    artifact_dir = tmp_path / "artifacts"
    train_frame = heart_frame.iloc[::3].reset_index(drop=True)
    validation = heart_frame.iloc[1::3].reset_index(drop=True)
    test = heart_frame.iloc[2::3].reset_index(drop=True)

    metrics = train(
        train_frame, artifact_dir, validation=validation, test=test, log_to_mlflow=False
    )

    assert metrics["cv_folds"] == 5
    assert 0 <= float(metrics["cv_accuracy"]) <= 1
    assert metrics["holdout"] == "test"
    assert metrics["test_rows"] == len(test)
    assert metrics["train_rows"] == len(train_frame)
    assert metrics["validation_rows"] == len(validation)

    bundle = load_bundle(artifact_dir / "heart_model.joblib")
    scored = predict(bundle, test[FEATURES])
    assert len(scored) == len(test)
    assert scored["prediction"].isin([0, 1]).all()


def test_resolve_inputs_rejects_mixed_sources(tmp_path, heart_frame):
    frame_path = tmp_path / "data.csv"
    heart_frame.to_csv(frame_path, index=False)
    parser = create_parser()
    args = parser.parse_args(
        ["--data", str(frame_path), "--train", str(tmp_path / "train.csv")]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_inputs(args)
