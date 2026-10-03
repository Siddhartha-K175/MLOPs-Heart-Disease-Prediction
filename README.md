# Heart disease MLOps website

This project is an educational end-to-end MVP:

- `src/train.py` validates a heart-disease CSV (own file or the bundled splits), runs 5-fold
  cross-validation, keeps a held-out test set, and writes model/metrics artifacts.
- Training logs parameters, metrics, and the model to MLflow, registers it as
  `heart-disease-classifier`, and moves the `champion` alias to the new version.
- `src/drift.py` compares new batches against the stored training distribution with PSI.
- `app.py` serves a Streamlit website with single-patient prediction and CSV batch scoring.
- `api.py` serves the same model over FastAPI with input range checks and batch limits.
- `artifacts/metrics.json` records the run, `heart_model.joblib` is the inference bundle, and
  `artifacts/reference.json` is the drift baseline.

## Run locally

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11+:

```powershell
uv sync --extra dev
uv run python -m src.train
uv run streamlit run app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`).

If `data/train.csv`, `data/validate.csv` and `data/test.csv` exist, training uses them
automatically (5-fold CV on train + validate, single evaluation on the untouched test split).
Otherwise it downloads the public UCI dataset and falls back to a random 80/20 split.

To view MLflow experiments and the Model Registry, run this in a second terminal:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://localhost:5000`, select the `heart-disease` experiment, or open the **Models**
page to see registered versions of `heart-disease-classifier` (the `champion` alias points at
the latest one).

## Training on your own data

```powershell
uv run python -m src.train --data data\your_heart_data.csv
uv run python -m src.train --train data\train.csv --validate data\validate.csv --test data\test.csv
```

The CSV must contain these columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal, target`. The target is converted to binary (`0` versus any
positive value). Rows with missing values are dropped, and rows outside the clinical ranges in
`src/model.py` (`FEATURE_BOUNDS`) are rejected instead of silently scored.

## REST API

`api.py` serves the trained model over FastAPI. Train first, then:

```powershell
uv run uvicorn api:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Service metadata |
| `GET /health` | Status plus training metrics (`cv_accuracy`, `data_hash`, `model_version`) |
| `POST /predict` | Score one patient (fields have defaults and enforced ranges) |
| `POST /predict-batch` | Score up to 1000 patient records |
| `POST /predict-csv` | Upload a CSV, get a CSV back with `prediction`, `risk_probability`, `risk_label` |

Invalid or out-of-range inputs return `422` instead of a prediction. Responses carry the
`model_version` that produced them.

## Drift monitoring

```powershell
uv run python -m src.drift --data data\new_batch.csv
uv run python -m src.drift --data data\new_batch.csv --fail-on-drift
```

Each feature gets a PSI score (`ok` < 0.1, `moderate` < 0.25, `significant` above). The
reference is rebuilt from train + validate every time you train. `--fail-on-drift` exits with
code 1 on significant drift so you can gate a pipeline on it. Batches under 100 rows are
flagged as unreliable.

## Test and lint

```powershell
uv run pytest
uv run ruff check .
```

Tests cover data validation, training (with and without splits), drift, and the API via
`TestClient`; they run offline with a synthetic frame. Lint uses `E`, `F`, `I`, `UP`, `B`.

Optional pre-commit hook:

```powershell
uv run pre-commit install
uv run pre-commit run --all-files
```

Continuous integration runs both commands on every push and pull request
(`.github/workflows/ci.yml`).

## Docker

Train first so `artifacts\heart_model.joblib` exists, then either a single container:

```powershell
docker build -t heart-disease-mlops .
docker run --rm -p 8501:8501 heart-disease-mlops
```

or the Streamlit site and API together:

```powershell
docker compose up --build
```

`docker compose up` serves the website on `http://localhost:8501` and the API docs on
`http://localhost:8000/docs`.

## X-ray scope

X-ray prediction is intentionally not wired to the tabular model. It needs a separately labeled
image dataset, a CNN/transfer-learning training pipeline, image preprocessing, dataset/license
review, and clinical validation. The UI marks this boundary rather than presenting an
unvalidated medical image classifier as reliable.

This demo is not medical advice and must not be used for diagnosis or treatment decisions.
