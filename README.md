# Heart disease MLOps website

This project is an educational end-to-end MVP:

- `src/train.py` validates a heart-disease CSV (own file or the bundled splits), drops
  duplicate rows, records a pre-cleaning missingness profile, runs 5-fold cross-validation,
  keeps a held-out test set, and writes model/metrics artifacts.
- Training logs parameters, metrics, and the model to MLflow, registers it as
  `heart-disease-classifier`, and moves the `champion` alias to the new version.
- `src/drift.py` compares new batches against the stored training distribution with PSI.
- `app.py` serves a Streamlit website with patient/admin sign-in, single-patient assessment
  and patient history (CSV batch scoring is available through the API).
- `api.py` serves the same model over FastAPI with input range checks and batch limits.
- `artifacts/metrics.json` records the run (accuracy, ROC-AUC, precision/recall/F1, confusion
  matrix, data profile), `artifacts/feature_importance.csv` holds the model coefficients,
  `heart_model.joblib` is the inference bundle, and `artifacts/reference.json` is the drift
  baseline.

## Run locally

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11+:

```powershell
uv sync --extra dev
uv run python -m src.train
uv run streamlit run app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`).

The **Single patient** tab stores every prediction in Supabase (see
[Database](#database-patients--history)). For records to go through the REST layer, also
start the API in a second terminal (`uv run uvicorn api:app --reload --port 8000`); if it
is not running, the UI scores locally and writes the same Supabase tables directly.

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
positive value). Rows with missing values and exact duplicate rows are dropped, and rows
outside the clinical ranges in `src/model.py` (`FEATURE_BOUNDS`) are rejected instead of
silently scored. Before cleaning, `src/profile.py` counts missing cells, rows with missing
values, and duplicates; the percentages land in `metrics.json` under `profile` and as
`profile_*` MLflow metrics.

### Class balance (optional SMOTE)

```powershell
uv run python -m src.train --smote
```

This inserts an imbalanced-learn `SMOTE` step after scaling, so each cross-validation fold
oversamples only its own training portion (no leakage into validation or test rows). The flag
is recorded in `metrics.json` (`smote`) and as an MLflow parameter. The default training stays
`class_weight="balanced"` without resampling.

### Evaluation outputs

Holdout `metrics.json` (mirrored in MLflow) includes `accuracy`, `roc_auc`, `precision`,
`recall`, `f1`, the 2x2 `confusion_matrix` plus `true_negatives`/`false_positives`/
`false_negatives`/`true_positives` scalars, CV means/stds, `top_feature`, and the data
`profile`. Coefficients for every model input are written to
`artifacts/feature_importance.csv` (`coefficient`, `abs_coefficient`, `odds_ratio`), sorted by
effect size, and logged to MLflow as an artifact.

### High-risk prevention guidance

When `risk_probability >= 0.5` the Streamlit app expands a prevention box, and
`POST /predict` returns a `guidance` block (`tier`, `headline`, `actions`,
`emergency_note`, `disclaimer`). Tiering: `high` >= 0.5, `moderate` >= 0.25, else `low`.
The advice mixes prompt-a-clinician steps, medication-*adherence* reminders, tips tied to the
patient's own elevated measurements (blood pressure, cholesterol, sugar, exercise symptoms,
vessel count), lifestyle basics, and an emergency warning-signs note (`src/advice.py`).
Batch/CSV responses stay lean (`guidance` is `null`). This is general health education only,
never a diagnosis or a prescription - a clinician decides medication.

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
| `POST /predict` | Score one patient (fields have defaults and enforced ranges), **store the record**, return `record_id` + `patient_id` |
| `POST /predict-batch` | Score up to 1000 patient records (stateless, not stored) |
| `POST /predict-csv` | Upload a CSV, get a CSV back with `prediction`, `risk_probability`, `risk_label` (stateless) |
| `POST /auth/register` | Patient self-sign-up (`username`, `password`, `name`): creates a new patient record and returns a token |
| `POST /auth/login` | Exchange `username` + `password` (admin or patient) for a bearer token (8-hour session) |
| `POST /auth/logout`, `GET /auth/me` | End the session / show the account's `role` and `patient_id` |
| `GET /me/history` 👤 | A signed-in patient's own record and history |
| `GET /patients` 🔒 | Every stored patient with prediction counts and latest score time |
| `GET /patients/{patient_id}` 🔒 | Profile row for one patient (`404` if unknown) |
| `GET /patients/{patient_id}/history` 🔒 | Full prediction history for one patient, newest first |
| `GET /predictions?limit=` 🔒 | Most recent predictions across all patients (1-200 rows) |

Invalid or out-of-range inputs return `422` instead of a prediction. Responses carry the
`model_version` that produced them. `POST /predict` also accepts three optional identity
fields - `patient_id`, `name`, `notes` - that link the record to a patient history row.

🔒 routes are **admin only** (`401` without a token, `403` for a patient token); 👤 needs a
patient token. Send `Authorization: Bearer <access_token>` from `/auth/login` or
`/auth/register`. `POST /predict` stays open for anonymous scoring; with a patient token the
record is always saved to that patient's own ID, whatever identity the body contains.

### Accounts: admins and patients

The website opens on a login page with **Patient** and **Admin** tabs.

- **Patients** create their own account (full name, username, password). Sign-up always
  creates a *new* patient record, so nobody can claim an existing Patient ID and read its
  history. Signed-in patients see **New assessment** (saved to their record) and **My history**.
- **Admins** are created by the system owner and see **New assessment** and **Patient
  records** for everyone, kept to the essentials: the patient list shows ID, name, number of
  assessments, latest risk and last assessed date; a patient's record shows each assessment's
  date, risk, result, BP, cholesterol and peak heart rate. Model metrics and service status
  are hidden from admins (status badges appear only when something is wrong). Batch CSV
  scoring is available through `POST /predict-csv`.

```powershell
uv run python -m src.auth create-admin admin                               # prompts for password
uv run python -m src.auth create-patient-login ravi --patient-id P-1001   # login for an existing record
uv run python -m src.auth list-accounts
uv run python -m src.auth delete-account ravi                             # medical record is kept
```

Usernames are case-insensitive (3-64 letters, digits, `.`, `-`, `_`); passwords need at
least 10 characters. Passwords are stored as salted scrypt hashes in `accounts`; each sign-in
creates a random bearer token whose SHA-256 is kept in `sessions` and expires after 8 hours.
Re-running `create-admin` resets that admin's password and signs it out everywhere. Databases
created before patient logins existed have their `admins` table migrated into `accounts`
automatically at startup (password hashes kept).

## Database (patients & history)

Every `POST /predict` is persisted to **Supabase** (Postgres), in the `heart_health` schema by
default, built from [`schema.sql`](schema.sql):

- **`patients`** - one row per person: `patient_id` (client-supplied, or auto-generated
  `P-XXXXXXXX` when only a name is given), `name`, `notes`, timestamps.
- **`predictions`** - one row per scored prediction: the 13 input features (with `CHECK`
  ranges mirroring `FEATURE_BOUNDS`), `prediction`, `risk_probability`, `risk_label`,
  `model_version`, `created_at`; linked to `patients.patient_id` (`NULL` = anonymous).
- **`patient_history`** (view) - predictions joined to patient identity, newest first.

### Set up Supabase

1. Create a project at [supabase.com](https://supabase.com) and note the database password.
2. In the project dashboard click **Connect** and copy the **Session pooler** connection
   string (it works over IPv4; the direct connection needs IPv6).
3. Copy [`.env.example`](.env.example) to `.env` and paste it as `SUPABASE_DB_URL`, with
   `[YOUR-PASSWORD]` replaced.

You do not have to run the DDL yourself: the API creates the `heart_health` schema and runs
`schema.sql` at startup (every statement is idempotent, so re-running is safe).

The tables live in their own `heart_health` schema, not `public`, so Supabase's
auto-generated REST API does not expose these health records. To see them in the
dashboard, open **Table Editor** and pick the `heart_health` schema, or query them in the
**SQL Editor**:

```sql
SELECT * FROM heart_health.patients;
SELECT * FROM heart_health.patient_history ORDER BY created_at DESC LIMIT 5;
SELECT COUNT(*) FROM heart_health.predictions;
```

### Configuration

Connection settings come from environment variables, optionally stored in a gitignored
[`.env`](.env) (copy [`.env.example`](.env.example) - real environment variables always
win over `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SUPABASE_DB_URL` | *(empty)* | **Postgres connection string from the Supabase dashboard** |
| `DB_SCHEMA` | `heart_health` | Schema for predictions and patients |
| `DB_TEST_SCHEMA` | `heart_health_test` | Throwaway schema the test suite drops and recreates |
| `HEART_API_URL` | `http://127.0.0.1:8000` | API base URL the Streamlit UI calls |

### From the command line: db.py

`db.py` runs one ad-hoc statement from PowerShell when you are not in the Supabase SQL Editor:

```powershell
uv run python db.py "SELECT patient_id, name FROM patients;"
uv run python db.py "UPDATE patients SET notes = 'reviewed' WHERE patient_id = 'P-1001';"
uv run python db.py --schema heart_health_test "SELECT COUNT(*) FROM predictions;"
```

The Streamlit **New assessment** tab sends measurements (and, for admins, identity) to
`POST /predict` with the user's token. Patients' **My history** renders `GET /me/history`;
the admin **Patient records** tab renders `GET /patients/{id}/history` and `GET /patients`. If the API is not
running, the UI falls back to scoring locally and writing the same Supabase tables directly.

## EDA

`src/eda.py` generates the exploratory chart set from the workflow:

```powershell
uv run python -m src.eda
uv run python -m src.eda --data data\your_raw.csv --output artifacts\eda
```

Outputs land in `artifacts/eda/`: `target_distribution.png`, `missing_values.png`,
`correlation_heatmap.png`, `boxplots.png`, `scatter_trends.png`, `groupby_target.png`,
plus `report.txt` (raw vs cleaned rows, columns with missing values, duplicate rows, target
counts, IQR outlier counts) and `missing_values.csv`. Needs the dev extra (`uv sync --extra dev`)
for matplotlib/seaborn; charts are ignored by git.

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

Tests cover data validation, profiling, training (splits, SMOTE, metrics, coefficients), EDA,
drift, and the API via `TestClient` with a synthetic frame. Database, CLI and API tests use
the `DB_TEST_SCHEMA` schema in the database at `SUPABASE_DB_URL` (dropped and recreated per
test) and are skipped when it is not set; CI runs them against a `postgres:15` service. Lint uses `E`,
`F`, `I`, `UP`, `B`.

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
`http://localhost:8000/docs`. Both containers connect to Supabase with
`SUPABASE_DB_URL` from `.env`, and the UI routes its writes through
`http://api:8000`, so predictions land in one database.

## X-ray scope

X-ray prediction is intentionally not wired to the tabular model. It needs a separately labeled
image dataset, a CNN/transfer-learning training pipeline, image preprocessing, dataset/license
review, and clinical validation, so the website deliberately offers no X-ray upload rather
than presenting an unvalidated medical image classifier as reliable.

This demo is not medical advice and must not be used for diagnosis or treatment decisions.
