"""Streamlit website for the heart-disease prediction model."""

from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd
import streamlit as st

from src import database
from src.advice import recommendations
from src.model import (
    FEATURE_BOUNDS,
    FEATURE_CHOICES,
    FEATURE_DEFAULTS,
    FEATURES,
    load_bundle,
    predict,
)

ROOT = Path(__file__).parent
ARTIFACT = ROOT / "artifacts" / "heart_model.joblib"
# The UI calls the API to score and store; when the API is down it falls back
# to scoring locally and writing the same Supabase tables directly.
API_URL = os.environ.get("HEART_API_URL", "http://127.0.0.1:8000").rstrip("/")
LOCAL_TZ = datetime.now().astimezone().tzinfo

# tier -> (title, accent colour, tint) for the result card and scale.
TIER_STYLE = {
    "low": ("Low risk", "#1E7B4F", "#E9F6EF"),
    "moderate": ("Moderate risk", "#A35F00", "#FFF5E3"),
    "high": ("High risk", "#B3132B", "#FDECEE"),
}

CSS = """
<style>
.block-container {padding-top: 1.25rem; max-width: 1240px;}
.app-header {display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: .75rem; padding: .9rem 1.2rem; margin-bottom: .75rem;
  background: #fff; border: 1px solid #E3E8EF; border-radius: 12px;}
.brand {font-size: 1.3rem; font-weight: 700; color: #1B2430; line-height: 1.2;}
.brand .mark {color: #C8102E;}
.brand-sub {font-size: .82rem; color: #5B6675;}
.pills {display: flex; gap: .4rem; flex-wrap: wrap;}
.pill {font-size: .74rem; padding: .22rem .65rem; border-radius: 999px;
  background: #EEF2F6; color: #334155; white-space: nowrap;}
.pill.ok::before {content: "\\25CF  "; color: #1E7B4F;}
.pill.warn::before {content: "\\25CF  "; color: #C27C00;}
[class*="st-key-card"] {background: #fff; border-radius: 12px;}
.section-title {font-weight: 650; font-size: .95rem; color: #1B2430;}
.section-sub {font-size: .8rem; color: #5B6675; margin-bottom: .2rem;}
.risk-card {border-radius: 12px; padding: 1.1rem 1.25rem; border: 1px solid;}
.risk-kicker {font-size: .8rem; color: #5B6675;}
.risk-value {font-size: 2.8rem; font-weight: 750; line-height: 1.1; margin: .15rem 0 .35rem;}
.risk-tier {display: inline-block; font-weight: 600; font-size: .78rem;
  padding: .18rem .65rem; border-radius: 999px; color: #fff;}
.risk-label {font-size: .9rem; color: #1B2430; margin-top: .5rem;}
.scale {position: relative; height: 10px; border-radius: 999px; margin: 1rem 0 .3rem;
  background: linear-gradient(90deg, #1E7B4F 0 25%, #E0A100 25% 50%, #B3132B 50% 100%);}
.scale-marker {position: absolute; top: -5px; width: 4px; height: 20px;
  background: #1B2430; border-radius: 2px; transform: translateX(-50%);}
.scale-labels {display: flex; font-size: .72rem; color: #5B6675; text-align: center;}
.flag {display: inline-block; font-size: .78rem; padding: .22rem .55rem;
  margin: 0 .3rem .35rem 0; border-radius: 6px; background: #FDECEE; color: #8A1022;}
.flag.ok {background: #E9F6EF; color: #14563A;}
.empty {text-align: center; padding: 2.2rem 1rem; color: #5B6675;}
.empty .icon {font-size: 2rem; color: #C8102E;}
.footer {font-size: .75rem; color: #7A8594; text-align: center; margin-top: 2rem;}
</style>
"""


def _call_api(
    method: str, path: str, payload: dict | None = None
) -> tuple[dict | list | None, str, bool]:
    """Call the FastAPI service. Returns (data, error_message, api_is_offline)."""
    try:
        response = httpx.request(method, f"{API_URL}{path}", json=payload, timeout=15.0)
    except httpx.HTTPError as error:
        return None, f"API at {API_URL} unreachable ({error.__class__.__name__})", True
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"API error {response.status_code}: {detail}", False
    try:
        return response.json(), "", False
    except ValueError:
        return None, "API returned a non-JSON response", False


@st.cache_data(ttl=30, show_spinner=False)
def _api_online() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


def _save_locally(values: dict[str, float], identity: dict[str, str | None],
                  row: pd.Series, model_version: str | None) -> dict:
    """Fallback writer used when the API is down (same tables, same schema)."""
    database.init_db()
    return database.save_prediction(
        features=values,
        prediction=int(row["prediction"]),
        risk_probability=float(row["risk_probability"]),
        risk_label=str(row["risk_label"]),
        model_version=model_version,
        **identity,
    )


def _load_history(patient_id: str) -> tuple[dict | None, str, bool]:
    """Fetch a patient's history from the API, falling back to Supabase directly."""
    data, error, offline = _call_api("GET", f"/patients/{quote(patient_id, safe='')}/history")
    if data is not None:
        return data, "", False
    if offline:
        try:
            local = database.get_patient_with_history(patient_id)
        except (database.DatabaseError, ValueError):
            local = None
        if local is not None:
            return local, f"{error} — read directly from Supabase.", True
        return None, f"No saved history in Supabase for this patient ({error}).", True
    return None, error, offline


def _pct(value: object, digits: int = 0) -> str:
    return f"{value:.{digits}%}" if isinstance(value, int | float) else "n/a"


def _local_times(column: pd.Series) -> pd.Series:
    return pd.to_datetime(column, utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)


def _section(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="section-title">{title}</div><div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def _number(column, feature: str, label: str, step: float, help: str | None = None) -> float:
    low, high = FEATURE_BOUNDS[feature]
    if step >= 1:
        value = column.number_input(
            label, min_value=int(low), max_value=int(high),
            value=int(FEATURE_DEFAULTS[feature]), step=int(step), help=help,
        )
    else:
        value = column.number_input(
            label, min_value=float(low), max_value=float(high),
            value=float(FEATURE_DEFAULTS[feature]), step=step, format="%.1f", help=help,
        )
    return float(value)


def _choice(column, feature: str, label: str, help: str | None = None,
            horizontal: bool = False) -> float:
    options = list(FEATURE_CHOICES[feature])
    index = options.index(int(FEATURE_DEFAULTS[feature]))
    describe = FEATURE_CHOICES[feature].get
    if horizontal:
        value = column.radio(
            label, options, index=index, format_func=describe, horizontal=True, help=help
        )
    else:
        value = column.selectbox(label, options, index=index, format_func=describe, help=help)
    return float(value)


def _findings(values: dict[str, float]) -> list[str]:
    """Plain-language list of the measurements that are outside usual targets."""
    found = []
    if values["trestbps"] >= 140:
        found.append(f"High resting BP ({values['trestbps']:.0f} mm Hg)")
    elif values["trestbps"] >= 130:
        found.append(f"Elevated resting BP ({values['trestbps']:.0f} mm Hg)")
    if values["chol"] >= 240:
        found.append(f"High cholesterol ({values['chol']:.0f} mg/dl)")
    elif values["chol"] >= 200:
        found.append(f"Borderline cholesterol ({values['chol']:.0f} mg/dl)")
    if values["fbs"] == 1:
        found.append("Fasting blood sugar > 120 mg/dl")
    if values["exang"] == 1:
        found.append("Angina during exercise")
    if values["oldpeak"] >= 2:
        found.append(f"ST depression {values['oldpeak']:.1f} mm")
    if values["slope"] in (2, 3):
        found.append(f"{FEATURE_CHOICES['slope'][int(values['slope'])]} ST segment")
    if values["ca"] >= 1:
        found.append(f"{int(values['ca'])} major vessel(s) narrowed")
    if values["thal"] in (6, 7):
        found.append(f"Thallium scan: {FEATURE_CHOICES['thal'][int(values['thal'])].lower()}")
    return found


def _assess(values: dict[str, float], identity: dict[str, str | None],
            metrics: dict) -> dict | None:
    """Score through the API (or locally when it is down) and return the result to show."""
    data, error, offline = _call_api("POST", "/predict", {**values, **identity})
    result = {
        "values": values,
        "assessed_at": datetime.now(),
        "record_id": None,
        "patient_id": None,
        "notice": "",
        "save_error": "",
    }
    if data is not None:
        probability = float(data["risk_probability"])
        result.update(
            probability=probability,
            risk_label=data["risk_label"],
            guidance=data["guidance"] or recommendations(probability, values),
            record_id=data.get("record_id"),
            patient_id=data.get("patient_id"),
        )
        return result
    if not offline:
        st.error(error)
        return None

    row = predict(bundle, pd.DataFrame([values])).iloc[0]
    probability = float(row["risk_probability"])
    result.update(
        probability=probability,
        risk_label=str(row["risk_label"]),
        guidance=recommendations(probability, values),
        notice=f"{error} — scored locally and saved directly to Supabase.",
    )
    version = metrics.get("model_version")
    try:
        saved = _save_locally(values, identity, row, str(version) if version is not None else None)
        result.update(record_id=saved["record_id"], patient_id=saved["patient_id"])
    except (database.DatabaseError, ValueError) as save_error:
        result["save_error"] = f"Could not save the record to Supabase: {save_error}"
    return result


def _render_result(result: dict) -> None:
    probability = result["probability"]
    guidance = result["guidance"]
    title, colour, tint = TIER_STYLE[guidance["tier"]]
    st.markdown(
        f"""
        <div class="risk-card" style="background:{tint};border-color:{colour}40">
          <div class="risk-kicker">Estimated probability of heart disease</div>
          <div class="risk-value" style="color:{colour}">{probability:.0%}</div>
          <span class="risk-tier" style="background:{colour}">{title}</span>
          <div class="scale"><div class="scale-marker" style="left:{probability * 100:.1f}%">
          </div></div>
          <div class="scale-labels"><span style="width:25%">Low &lt;25%</span>
            <span style="width:25%">Moderate</span><span style="width:50%">High &ge;50%</span>
          </div>
          <div class="risk-label">{html.escape(result["risk_label"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    when = result["assessed_at"].strftime("%d %b %Y, %H:%M")
    if result["record_id"] is not None:
        owner = (
            f"patient **{result['patient_id']}**" if result["patient_id"] else "an anonymous record"
        )
        st.caption(f"Saved to {owner} · record #{result['record_id']} · {when}")
    if result["notice"]:
        st.warning(result["notice"])
    if result["save_error"]:
        st.error(result["save_error"])

    findings = _findings(result["values"])
    st.markdown('<div class="section-title">Findings to review</div>', unsafe_allow_html=True)
    chips = (
        "".join(f'<span class="flag">{html.escape(item)}</span>' for item in findings)
        if findings
        else '<span class="flag ok">No measurements outside usual targets</span>'
    )
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

    if guidance["tier"] == "low":
        st.caption(guidance["disclaimer"])
        return
    label = (
        "Prevention & next steps" if guidance["tier"] == "high" else "Risk-reduction tips"
    )
    with st.expander(label, expanded=guidance["tier"] == "high"):
        st.markdown(f"**{guidance['headline']}**")
        for action in guidance["actions"]:
            st.markdown(f"- {action}")
        if guidance.get("emergency_note"):
            st.error(guidance["emergency_note"])
        st.caption(guidance["disclaimer"])


st.set_page_config(page_title="CardioRisk · Heart Health", page_icon="❤️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

try:
    bundle = load_bundle(ARTIFACT)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.code("uv run python -m src.train\nuv run streamlit run app.py")
    st.stop()

metrics = bundle.get("metrics", {})
api_online = _api_online()
pills = [
    ("ok", "API online") if api_online else ("warn", "API offline · scoring locally"),
    (
        ("ok", f"Supabase · {database.DB_SCHEMA}")
        if database.DB_URL
        else ("warn", "Database not configured")
    ),
    ("ok", f"Model v{metrics.get('model_version', '?')}"),
]
st.markdown(
    f"""
    <div class="app-header">
      <div>
        <div class="brand"><span class="mark">&#10084;</span> CardioRisk</div>
        <div class="brand-sub">
          Coronary heart-disease risk assessment · clinical decision support
        </div>
      </div>
      <div class="pills">
        {"".join(f'<span class="pill {kind}">{html.escape(text)}</span>' for kind, text in pills)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Model")
    left, right = st.columns(2)
    left.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}" if "roc_auc" in metrics else "n/a")
    right.metric("Accuracy", _pct(metrics.get("accuracy")))
    left.metric("Recall", _pct(metrics.get("recall")))
    right.metric("Precision", _pct(metrics.get("precision")))
    st.caption(
        f"Version {metrics.get('model_version', 'n/a')} · "
        f"{metrics.get('dataset_rows', 'n/a')} training patients · "
        f"CV accuracy {_pct(metrics.get('cv_accuracy'))} ± {_pct(metrics.get('cv_accuracy_std'))}"
    )
    st.caption(f"Data hash `{metrics.get('data_hash', 'n/a')}`")
    st.divider()
    st.markdown("### About")
    st.caption(
        "Logistic-regression model trained on the UCI Cleveland heart-disease study. It "
        "estimates the probability of angiographic coronary artery disease from 13 routine "
        "clinical measurements."
    )
    st.warning(
        "Decision support only — not a diagnosis. Always confirm with a qualified clinician."
    )

tab_single, tab_history, tab_batch, tab_xray = st.tabs(
    ["New assessment", "Patient history", "Batch scoring", "Chest X-ray"]
)

with tab_single:
    form_column, result_column = st.columns([3, 2], gap="large")
    with form_column:
        with st.form("assessment", border=False):
            values: dict[str, float] = {}
            with st.container(border=True, key="card_patient"):
                _section("Patient", "Optional — link this assessment to a patient record.")
                left, right = st.columns(2)
                patient_id = left.text_input("Patient ID", placeholder="e.g. P-1001")
                patient_name = right.text_input("Full name", placeholder="e.g. Asha Rao")
                patient_note = st.text_input(
                    "Clinical notes", placeholder="e.g. referred for chest discomfort on exertion"
                )

            with st.container(border=True, key="card_vitals"):
                _section("Symptoms & vitals", "Demographics, presenting symptoms and bloods.")
                left, right = st.columns(2)
                values["age"] = _number(left, "age", "Age (years)", 1)
                values["sex"] = _choice(right, "sex", "Sex", horizontal=True)
                values["cp"] = _choice(
                    left, "cp", "Chest-pain type",
                    help="Typical angina: substernal pain brought on by exertion and relieved "
                    "by rest or nitroglycerin.",
                )
                values["trestbps"] = _number(
                    right, "trestbps", "Resting blood pressure (mm Hg)", 1,
                    help="Systolic, on admission. Normal < 120; hypertension ≥ 140.",
                )
                values["chol"] = _number(
                    left, "chol", "Serum cholesterol (mg/dl)", 1,
                    help="Desirable < 200; borderline 200–239; high ≥ 240.",
                )
                values["fbs"] = _choice(
                    right, "fbs", "Fasting blood sugar (mg/dl)", horizontal=True
                )

            with st.container(border=True, key="card_exercise"):
                _section("ECG & exercise stress test", "Resting ECG and treadmill test results.")
                left, right = st.columns(2)
                values["restecg"] = _choice(left, "restecg", "Resting ECG")
                values["thalach"] = _number(
                    right, "thalach", "Peak heart rate (bpm)", 1,
                    help="Maximum heart rate reached during the test. Age-predicted maximum "
                    "is about 220 − age.",
                )
                values["exang"] = _choice(
                    left, "exang", "Exercise-induced angina", horizontal=True
                )
                values["oldpeak"] = _number(
                    right, "oldpeak", "ST depression (mm)", 0.1,
                    help="ST-segment depression induced by exercise relative to rest.",
                )
                values["slope"] = _choice(left, "slope", "Peak-exercise ST slope")

            with st.container(border=True, key="card_imaging"):
                _section("Imaging", "Fluoroscopy and thallium perfusion scan.")
                left, right = st.columns(2)
                values["ca"] = _choice(
                    left, "ca", "Major vessels narrowed",
                    help="Number of major coronary vessels coloured by fluoroscopy.",
                )
                values["thal"] = _choice(
                    right, "thal", "Thallium stress scan",
                    help="Fixed defect: scarred tissue. Reversible defect: reduced blood flow "
                    "under stress.",
                )

            submitted = st.form_submit_button("Calculate risk", type="primary", width="stretch")

    if submitted:
        identity = {
            "patient_id": patient_id.strip() or None,
            "name": patient_name.strip() or None,
            "notes": patient_note.strip() or None,
        }
        ordered = {feature: values[feature] for feature in FEATURES}
        with st.spinner("Scoring and saving…"):
            outcome = _assess(ordered, identity, metrics)
        if outcome is not None:
            st.session_state["latest_assessment"] = outcome
            if outcome["patient_id"]:
                st.session_state["last_patient_id"] = outcome["patient_id"]

    with result_column:
        assessment = st.session_state.get("latest_assessment")
        if assessment:
            _render_result(assessment)
        else:
            with st.container(border=True, key="card_empty"):
                st.markdown(
                    '<div class="empty"><div class="icon">&#10084;</div>'
                    "<b>No assessment yet</b><br>Complete the form and press "
                    "<b>Calculate risk</b> to see the estimated probability, the findings "
                    "behind it and suggested next steps.</div>",
                    unsafe_allow_html=True,
                )

with tab_history:
    last_patient = st.session_state.get("last_patient_id", "")
    search_column, _ = st.columns([2, 3])
    typed = search_column.text_input(
        "Find a patient", placeholder=last_patient or "Patient ID, e.g. P-1001",
        key="history_lookup",
    ).strip()
    lookup_id = typed or last_patient
    if lookup_id:
        record, note, was_offline = _load_history(lookup_id)
        if record and record["history"]:
            patient = record["patient"]
            history = pd.DataFrame(record["history"])
            history["created_at"] = _local_times(history["created_at"])
            history = history.sort_values("created_at", ascending=False)
            latest = float(history["risk_probability"].iloc[0])
            previous = (
                float(history["risk_probability"].iloc[1]) if len(history) > 1 else None
            )

            with st.container(border=True, key="card_patient_summary"):
                st.markdown(f"#### {html.escape(patient.get('name') or patient['patient_id'])}")
                caption = f"Patient ID `{patient['patient_id']}`"
                if patient.get("notes"):
                    caption += f" · {patient['notes']}"
                st.caption(caption)
                first, second, third = st.columns(3)
                first.metric("Assessments", record["count"])
                second.metric(
                    "Latest risk",
                    f"{latest:.0%}",
                    delta=f"{(latest - previous) * 100:+.0f} pts" if previous is not None else None,
                    delta_color="inverse",
                )
                third.metric("First assessed", history["created_at"].min().strftime("%d %b %Y"))
            if note:
                st.caption(note)

            if len(history) > 1:
                trend = history.set_index("created_at")[["risk_probability"]] * 100
                st.line_chart(
                    trend.rename(columns={"risk_probability": "Risk (%)"}),
                    y_label="Risk (%)", height=220,
                )
            st.dataframe(
                history[
                    ["created_at", "risk_probability", "risk_label", "trestbps", "chol",
                     "thalach", "model_version"]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "created_at": st.column_config.DatetimeColumn(
                        "Assessed", format="D MMM YYYY, HH:mm"
                    ),
                    "risk_probability": st.column_config.ProgressColumn(
                        "Risk", min_value=0.0, max_value=1.0, format="percent"
                    ),
                    "risk_label": "Result",
                    "trestbps": st.column_config.NumberColumn("BP (mm Hg)", format="%d"),
                    "chol": st.column_config.NumberColumn("Cholesterol", format="%d"),
                    "thalach": st.column_config.NumberColumn("Peak HR", format="%d"),
                    "model_version": "Model",
                },
            )
        elif record:
            st.info("This patient exists but has no saved assessments yet.")
        elif "404" in note or "No saved history" in note:
            st.info(f"No patient found with ID {lookup_id!r}. Run an assessment first.")
        else:
            st.info(f"Could not load history for {lookup_id!r}: {note}")
    else:
        st.caption("Search by Patient ID, or run an assessment with an ID or name first.")

    with st.expander("All patients"):
        patients_data, patients_error, patients_offline = _call_api("GET", "/patients")
        if patients_data is None and patients_offline:
            try:
                rows = database.list_patients()
            except (database.DatabaseError, ValueError):
                rows = None
            if rows is not None:
                patients_data = {"patients": rows, "count": len(rows)}
                st.caption(f"{patients_error} — read directly from Supabase.")
        if patients_data is not None and patients_data["patients"]:
            directory = pd.DataFrame(patients_data["patients"])
            for column in ("created_at", "last_prediction_at"):
                directory[column] = _local_times(directory[column])
            st.dataframe(
                directory[
                    ["patient_id", "name", "prediction_count", "last_prediction_at", "notes"]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "patient_id": "Patient ID",
                    "name": "Name",
                    "prediction_count": st.column_config.NumberColumn("Assessments"),
                    "last_prediction_at": st.column_config.DatetimeColumn(
                        "Last assessed", format="D MMM YYYY, HH:mm"
                    ),
                    "notes": "Notes",
                },
            )
        elif patients_data is not None:
            st.info("No patients stored yet.")
        else:
            st.info(f"Could not list patients: {patients_error}")

with tab_batch:
    st.markdown("#### Batch scoring")
    st.caption(
        "Score many patients at once from a CSV containing the 13 model columns. "
        "Batch results are not saved to patient records."
    )
    template = pd.DataFrame([FEATURE_DEFAULTS])[FEATURES]
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False),
        file_name="cardiorisk_batch_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            results = predict(bundle, batch)
            combined = pd.concat(
                [batch.reset_index(drop=True), results.reset_index(drop=True)], axis=1
            )
            risk = combined["risk_probability"]
            first, second, third = st.columns(3)
            first.metric("Patients scored", len(combined))
            second.metric("High risk (≥ 50%)", int((risk >= 0.5).sum()))
            third.metric("Average risk", f"{risk.mean():.0%}")
            st.dataframe(
                combined,
                hide_index=True,
                width="stretch",
                column_config={
                    "risk_probability": st.column_config.ProgressColumn(
                        "Risk", min_value=0.0, max_value=1.0, format="percent"
                    ),
                    "risk_label": "Result",
                    "prediction": "Prediction",
                },
            )
            st.download_button(
                "Download results",
                combined.to_csv(index=False),
                file_name="cardiorisk_predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not score this file: {error}")

with tab_xray:
    st.markdown("#### Chest X-ray analysis")
    st.info(
        "Not available yet. A chest X-ray is an image modality and needs a separately "
        "trained, clinically validated model (for example on NIH ChestX-ray14 or CheXpert). "
        "An image classifier would never replace review by a radiologist or cardiologist."
    )
    st.file_uploader(
        "X-ray image (reserved for a future model)",
        type=["png", "jpg", "jpeg"],
        disabled=True,
    )

st.markdown(
    '<div class="footer">CardioRisk is an educational MLOps project. Estimates come from a '
    "statistical model and are not a medical diagnosis.</div>",
    unsafe_allow_html=True,
)
