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
uv sync --all-extras
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

### Multi-hospital data and external validation

The UCI collection has four hospitals; the production model uses only Cleveland.
`src/multisite.py` loads all four (920 patients), keeps gaps as missing values for the
pipeline's imputers (placeholder zeros and out-of-range values are blanked rather than
rejected), and validates across hospitals:

```powershell
uv run python -m src.multisite                      # artifacts/site_validation.json + MLflow run
uv run python -m src.train --multisite              # candidate trained on all four hospitals
uv run python -m src.train --multisite --core-features   # ...without ca, thal, slope
```

- **Cleveland -> other hospitals**: train on Cleveland (as production does), test on each
  other hospital. **Leave-one-site-out**: train on three hospitals, test on the fourth.
- Both run for all 13 features and a **core** set without `ca`, `thal` and `slope`, which
  most non-Cleveland patients never had measured (90-99% missing).
- No missing-value indicator columns: missingness mostly reveals the hospital, and disease
  rates differ by hospital (36-94%), so indicators would teach the model the hospital.

Findings (logistic regression, ROC-AUC / recall at the 0.5 threshold):

| Trained on Cleveland, tested on | 13 features | core 10 features |
| --- | --- | --- |
| Cleveland (internal 5-fold CV) | 0.92 / - | 0.88 / - |
| Hungary (n=293, 36% disease) | 0.88 / 0.66 | 0.88 / 0.79 |
| Switzerland (n=123, 94%) | 0.76 / 0.62 | 0.76 / 0.82 |
| VA Long Beach (n=199, 74%) | 0.70 / 0.52 | 0.74 / 0.86 |

The 13-feature model looks best at home but transfers worst: it leans on the fluoroscopy
and thallium results other hospitals lack, misses about half of VA patients with disease
and under-predicts risk there (45% predicted vs 74% observed). The core model gives up a
little at home for far better recall and calibration elsewhere. Candidates trained on all
four hospitals are registered separately (`heart-disease-classifier-multisite-all` /
`-core`) and do not replace the production `champion`.

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
| `POST /predict` | Score one patient (fields have defaults and enforced ranges), **store the record** with its `assessed_on` test date (default today, never in the future), return `record_id` + `patient_id` |
| `POST /predict-batch` | Score up to 1000 patient records (stateless, not stored) |
| `POST /predict-csv` | Upload a CSV, get a CSV back with `prediction`, `risk_probability`, `risk_label` (stateless) |
| `POST /auth/register` | Patient email sign-up (`email`, `password`, `name`) through Supabase Auth, which emails a confirmation link (`202`) |
| `POST /auth/resend-confirmation` | Send the confirmation email again (`email`) |
| `POST /auth/login` | `username` (admin) or patient email + `password` for a bearer token (8-hour session); `403` until the email is confirmed |
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

- **Patients register with their email** through Supabase Auth. Supabase sends a
  confirmation link and keeps the password; sign-in is refused until the link is opened.
  The patient's record (new Patient ID, name, registration date) is created on their first
  confirmed sign-in, so unconfirmed sign-ups never leave empty records, and nobody can claim
  an existing Patient ID. Signed-in patients see **New assessment** (saved to their record)
  and **My history**.
- **Admins** sign in with a username created by the system owner and see **New assessment**
  and **Patient records** for everyone, kept to the essentials: the patient list shows ID,
  name, number of assessments, latest risk, last test date and registration date; a
  patient's record shows each assessment's test date, risk, result, BP, cholesterol and peak
  heart rate. Model metrics are hidden from admins, and the page header shows only the
  CardioRisk name (no service-status tags). Batch CSV scoring is available through
  `POST /predict-csv`.

Every assessment stores the **test date** entered on the form (`predictions.assessed_on`,
default today) separately from when it was saved (`created_at`); history, trends and "last
test" use the test date. A patient's **registration date** is `patients.created_at`.

Email sign-up settings (`.env`, see [`.env.example`](.env.example)):

| Variable | Purpose |
| --- | --- |
| `SUPABASE_URL` | `https://<project-ref>.supabase.co` |
| `SUPABASE_PUBLISHABLE_KEY` | Dashboard -> Project Settings -> API Keys -> **publishable** key (safe to share; never the secret key) |
| `APP_URL` | Where the confirmation link sends patients back (default `http://localhost:8502`); add it under Authentication -> URL Configuration |

Supabase's built-in email service only delivers to addresses of your Supabase organisation's
members and is rate limited; for real patients, set up custom SMTP under
Authentication -> Emails -> SMTP Settings.

```powershell
uv run python -m src.auth create-admin admin          # prompts for password
uv run python -m src.auth list-accounts
uv run python -m src.auth delete-account someone@example.com   # medical record is kept
```

Admin usernames are case-insensitive (3-64 letters, digits, `.`, `-`, `_`); passwords need
at least 10 characters and are stored as salted scrypt hashes in `accounts`. Each sign-in
creates a random bearer token whose SHA-256 is kept in `sessions` and expires after 8 hours.
Re-running `create-admin` resets that admin's password and signs it out everywhere. Older
databases are migrated automatically at startup: the earlier `admins` table moves into
`accounts` (password hashes kept), and existing assessments get a test date taken from when
they were saved.

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
counts, IQR outlier counts) and `missing_values.csv`. Needs the dev extra (`uv sync --all-extras`)
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

## Deploy the API to Vercel

The FastAPI backend (`api.py`) deploys to Vercel as one serverless function.
`[tool.vercel] entrypoint = "api:app"` in `pyproject.toml` points Vercel at it (otherwise it
would pick `app.py`, the Streamlit site). Vercel installs only the core dependencies; training
(`train` extra: MLflow, imbalanced-learn) and the website (`ui` extra: Streamlit) are left out,
which keeps the bundle around 250 MB against Vercel's 500 MB limit. `vercel.json` runs the
function in Dublin (`dub1`), next to the Supabase database, and excludes tests and data from
the bundle. The trained model `artifacts/heart_model.joblib` is committed so the deployment can
load it; retrain and commit it again to ship a new model version.

1. Push the repository to GitHub, then on vercel.com choose **Add New -> Project** and import it.
2. Under **Environment Variables** add `SUPABASE_DB_URL`, `DB_SCHEMA`, `SUPABASE_URL`,
   `SUPABASE_PUBLISHABLE_KEY` and `APP_URL` (same meaning as in `.env`).
3. Deploy, then open `https://<project>.vercel.app/health` and `/docs`.

## Deploy the website to Render

The Streamlit website runs on Render as a Docker web service (Vercel cannot host it: it
needs a long-running server with WebSocket connections). Render shows the page without
host branding, unlike Streamlit Community Cloud, which adds Fork/GitHub buttons and a
creator badge around apps from public repositories.

1. On render.com choose **New -> Blueprint**, connect this repository and apply
   [`render.yaml`](render.yaml): a free `cardiorisk` web service in Frankfurt, built from the
   `Dockerfile`, with `HEART_API_URL` pointing at the Vercel API.
2. That is the only setting the site needs: it scores, saves and signs users in through the
   API, so the database password never has to be stored on the website host.
3. Set `APP_URL` on Vercel and the Supabase Auth **Site URL** (plus a redirect URL) to the
   site's `https://<service>.onrender.com` address so confirmation emails return to it.

Free Render services sleep after about 15 minutes without traffic; the next visit takes
roughly a minute while the container starts. Pushes to `main` redeploy automatically.

The `Dockerfile` installs [`website/requirements.txt`](website/requirements.txt), pinned
from `uv.lock`, so the container runs the same scikit-learn the model was trained with.
Regenerate it after dependency changes:
`uv export --frozen --extra ui --no-dev --no-emit-project --no-hashes -o website/requirements.txt`.
`website/streamlit_app.py` is an alternative entrypoint for Streamlit Community Cloud.

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
