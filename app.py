"""Streamlit website for the heart-disease prediction model."""

from __future__ import annotations

import html
import os
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd
import streamlit as st

from src import auth, database, supabase_auth
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
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background-color: #F3F5F8 !important; border: 1px solid #D5DCE4 !important;}
[data-testid="stTextInputRootElement"] > div, [data-testid="stNumberInputContainer"] > div,
[data-testid="stTextInputField"], [data-testid="stNumberInputField"] {
  background-color: transparent !important;}
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
.login-hero {text-align: center; margin: 1.5rem 0 1rem;}
.login-hero .icon {font-size: 2.4rem; color: #C8102E; line-height: 1;}
.login-hero h2 {margin: .4rem 0 .2rem; font-size: 1.6rem; color: #1B2430;}
.login-hero p {color: #5B6675; font-size: .92rem; margin: 0;}
</style>
"""


def _call_api(
    method: str, path: str, payload: dict | None = None, token: str | None = None
) -> tuple[dict | list | None, str, bool]:
    """Call the FastAPI service. Returns (data, error_message, api_is_offline)."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        response = httpx.request(
            method, f"{API_URL}{path}", json=payload, headers=headers, timeout=15.0
        )
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
                  row: pd.Series, model_version: str | None, assessed_on: date) -> dict:
    """Fallback writer used when the API is down (same tables, same schema)."""
    database.init_db()
    return database.save_prediction(
        features=values,
        prediction=int(row["prediction"]),
        risk_probability=float(row["risk_probability"]),
        risk_label=str(row["risk_label"]),
        model_version=model_version,
        assessed_on=assessed_on,
        **identity,
    )


def _account_from(session: dict, token: str) -> dict:
    """Session-state record for a signed-in user."""
    return {
        "username": session["username"],
        "role": session["role"],
        "patient_id": session.get("patient_id"),
        "name": session.get("name"),
        "email": session.get("email"),
        "token": token,
    }


def _api_detail(error: str) -> str:
    return error.split(": ", 1)[-1] if error.startswith("API error") else error


def _sign_in(identifier: str, password: str, role: str) -> tuple[dict | None, str]:
    """Sign in through the API, or directly against Supabase when it is down.

    Patients use their email (checked by Supabase Auth); admins use a username.
    """
    wrong = "Incorrect email or password." if role == "patient" else (
        "Incorrect username or password."
    )
    data, error, offline = _call_api(
        "POST", "/auth/login", {"username": identifier, "password": password}
    )
    if data is not None:
        token = data["access_token"]
        profile, _, _ = _call_api("GET", "/auth/me", token=token)
        account = _account_from({**data, **(profile or {})}, token)
    elif not offline:
        return None, wrong if "API error 401" in error else _api_detail(error)
    else:
        try:
            session = auth.login(identifier, password)
            profile = auth.current_session(session["access_token"]) if session else None
        except auth.SignInError as refusal:
            return None, str(refusal)
        except (database.DatabaseError, ValueError) as local_error:
            return None, f"{error}; Supabase sign-in also failed: {local_error}"
        if session is None:
            return None, wrong
        account = _account_from({**session, **(profile or {})}, session["access_token"])

    if account["role"] != role:
        # Right password, wrong door: do not leave the session open.
        _end_session(account["token"])
        other = "Admin" if account["role"] == "admin" else "Patient"
        return None, f"This is a {other.lower()} account. Use the {other} tab to sign in."
    return account, ""


def _register(email: str, password: str, name: str) -> tuple[str | None, str]:
    """Patient email sign-up; Supabase emails a confirmation link. Returns the email."""
    data, error, offline = _call_api(
        "POST", "/auth/register", {"email": email, "password": password, "name": name}
    )
    if data is not None:
        return data["email"], ""
    if not offline:
        return None, _api_detail(error)
    try:
        return auth.register_patient(email, password, name), ""
    except ValueError as local_error:
        return None, str(local_error)


def _resend_confirmation(email: str) -> str:
    _, error, offline = _call_api("POST", "/auth/resend-confirmation", {"email": email})
    if offline:
        try:
            supabase_auth.resend_confirmation(auth.normalize_email(email))
            error = ""
        except (ValueError, supabase_auth.AuthError) as local_error:
            error = str(local_error)
    return _api_detail(error)


def _end_session(token: str) -> None:
    _, _, offline = _call_api("POST", "/auth/logout", token=token)
    if offline:
        try:
            auth.logout(token)
        except (database.DatabaseError, ValueError):
            pass


def _sign_out() -> None:
    account = st.session_state.pop("account", None)
    for key in ("latest_assessment", "last_patient_id"):
        st.session_state.pop(key, None)
    if account:
        _end_session(account["token"])


def _signed_in_read(
    path: str, read_locally, token: str, role: str
) -> tuple[dict | None, str, bool]:
    """GET a signed-in API route; when the API is down, check the session and read
    Supabase directly. Returns (data, note, session_expired)."""
    data, error, offline = _call_api("GET", path, token=token)
    if data is not None:
        return data, "", False
    if not offline:
        return None, error, "API error 401" in error
    try:
        session = auth.current_session(token)
        if session is None or session["role"] != role:
            return None, "Your session has expired.", True
        local = read_locally()
    except (database.DatabaseError, ValueError) as local_error:
        return None, f"{error}; Supabase read also failed: {local_error}", False
    if local is None:
        return None, f"404 — not found in Supabase ({error}).", False
    return local, f"{error} — read directly from Supabase.", False


def _expire_session() -> None:
    st.session_state.pop("account", None)
    st.session_state["session_expired"] = True
    st.rerun()


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
            metrics: dict, token: str, assessed_on: date) -> dict | None:
    """Score through the API (or locally when it is down) and return the result to show."""
    payload = {**values, **identity, "assessed_on": assessed_on.isoformat()}
    data, error, offline = _call_api("POST", "/predict", payload, token=token)
    result = {
        "values": values,
        "assessed_at": datetime.now(),
        "assessed_on": assessed_on,
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
        saved = _save_locally(
            values, identity, row, str(version) if version is not None else None, assessed_on
        )
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

    tested = result["assessed_on"].strftime("%d %b %Y")
    if result["record_id"] is not None:
        owner = (
            f"patient **{result['patient_id']}**" if result["patient_id"] else "an anonymous record"
        )
        st.caption(f"Test date {tested} · saved to {owner} · record #{result['record_id']}")
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


def _render_record(record: dict, note: str, *, own: bool) -> None:
    """A patient's summary, risk trend and assessment table."""
    patient = record["patient"]
    if not record["history"]:
        message = (
            "You have no saved assessments yet. Run your first one on the "
            "**New assessment** tab."
            if own
            else "This patient exists but has no saved assessments yet."
        )
        st.info(message)
        return
    history = pd.DataFrame(record["history"])
    history["created_at"] = _local_times(history["created_at"])
    history["assessed_on"] = pd.to_datetime(history["assessed_on"])
    history = history.sort_values(["assessed_on", "created_at"], ascending=False)
    latest = float(history["risk_probability"].iloc[0])
    previous = float(history["risk_probability"].iloc[1]) if len(history) > 1 else None

    with st.container(border=True, key="card_patient_summary"):
        st.markdown(f"#### {html.escape(patient.get('name') or patient['patient_id'])}")
        caption = f"Patient ID `{patient['patient_id']}`"
        if patient.get("notes") and not own:
            caption += f" · {patient['notes']}"
        st.caption(caption)
        registered = _local_times(pd.Series([patient["created_at"]])).iloc[0]
        first, second, third, fourth = st.columns(4)
        first.metric("Assessments", record["count"])
        second.metric(
            "Latest risk",
            f"{latest:.0%}",
            delta=f"{(latest - previous) * 100:+.0f} pts" if previous is not None else None,
            delta_color="inverse",
        )
        third.metric("Last test", history["assessed_on"].iloc[0].strftime("%d %b %Y"))
        fourth.metric("Registered", registered.strftime("%d %b %Y"))
    if note:
        st.caption(note)

    if len(history) > 1:
        trend = history.set_index("assessed_on")[["risk_probability"]] * 100
        st.line_chart(
            trend.rename(columns={"risk_probability": "Risk (%)"}),
            y_label="Risk (%)", height=220,
        )
    st.dataframe(
        history[["assessed_on", "risk_probability", "risk_label", "trestbps", "chol", "thalach"]],
        hide_index=True,
        width="stretch",
        column_config={
            "assessed_on": st.column_config.DateColumn("Test date", format="D MMM YYYY"),
            "risk_probability": st.column_config.ProgressColumn(
                "Risk", min_value=0.0, max_value=1.0, format="percent"
            ),
            "risk_label": "Result",
            "trestbps": st.column_config.NumberColumn("BP (mm Hg)", format="%d"),
            "chol": st.column_config.NumberColumn("Cholesterol", format="%d"),
            "thalach": st.column_config.NumberColumn("Peak HR", format="%d"),
        },
    )


def _sign_in_form(role: str) -> None:
    is_patient = role == "patient"
    with st.form(f"{role}_sign_in", border=False):
        identifier = st.text_input(
            "Email" if is_patient else "Username",
            autocomplete="email" if is_patient else "username",
            placeholder="you@example.com" if is_patient else None,
            key=f"{role}_user",
        )
        password = st.text_input(
            "Password", type="password", autocomplete="current-password", key=f"{role}_pass"
        )
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
    if submitted:
        if not identifier.strip() or not password:
            st.error(f"Enter your {'email' if is_patient else 'username'} and password.")
            return
        with st.spinner("Signing in…"):
            account, error = _sign_in(identifier, password, role)
        if account:
            st.session_state["account"] = account
            st.rerun()
        st.error(error)
        if is_patient and "confirm your email" in error:
            st.session_state["unconfirmed_email"] = identifier.strip()
    unconfirmed = st.session_state.get("unconfirmed_email") if is_patient else None
    if unconfirmed and st.button("Resend confirmation email", key="resend_from_sign_in"):
        problem = _resend_confirmation(unconfirmed)
        if problem:
            st.error(problem)
        else:
            st.success(f"We sent a new confirmation link to {unconfirmed}.")


def _register_form() -> None:
    sent_to = st.session_state.get("registered_email")
    if sent_to:
        st.success(
            f"Check your inbox at **{sent_to}** and open the confirmation link. "
            "Then come back and sign in on the **Sign in** tab."
        )
        left, right = st.columns(2)
        if left.button("Resend email", key="resend_after_register", width="stretch"):
            problem = _resend_confirmation(sent_to)
            if problem:
                st.error(problem)
            else:
                st.info("Sent again. It can take a minute to arrive; check your spam folder.")
        if right.button("Use another email", key="register_again", width="stretch"):
            st.session_state.pop("registered_email", None)
            st.rerun()
        return

    with st.form("patient_register", border=False):
        name = st.text_input("Full name", placeholder="e.g. Asha Rao")
        email = st.text_input("Email", placeholder="you@example.com", autocomplete="email")
        password = st.text_input(
            "Choose a password", type="password", autocomplete="new-password",
            help=f"At least {auth.MIN_PASSWORD_LENGTH} characters.",
        )
        confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
        consent = st.checkbox(
            "I understand CardioRisk gives a statistical estimate, not a medical diagnosis."
        )
        submitted = st.form_submit_button("Create account", type="primary", width="stretch")
    if not submitted:
        return
    if not name.strip() or not email.strip():
        st.error("Enter your full name and email.")
    elif len(password) < auth.MIN_PASSWORD_LENGTH:
        st.error(f"Password must be at least {auth.MIN_PASSWORD_LENGTH} characters.")
    elif password != confirm:
        st.error("The two passwords do not match.")
    elif not consent:
        st.error("Please confirm you understand this is not a diagnosis.")
    else:
        with st.spinner("Creating your account…"):
            registered, error = _register(email, password, name)
        if registered:
            st.session_state["registered_email"] = registered
            st.rerun()
        st.error(error)


def _render_login() -> None:
    _, middle, _ = st.columns([1, 1.6, 1])
    with middle:
        st.markdown(
            '<div class="login-hero"><div class="icon">&#10084;</div>'
            "<h2>Welcome to CardioRisk</h2>"
            "<p>Sign in to assess heart-disease risk and follow your results over time.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.pop("session_expired", False):
            st.warning("Your session has expired. Please sign in again.")
        if st.session_state.pop("signed_out", False):
            st.success("You have been signed out.")
        with st.container(border=True, key="card_login"):
            patient_tab, admin_tab = st.tabs(["Patient", "Admin"])
            with patient_tab:
                sign_in_tab, register_tab = st.tabs(["Sign in", "Create account"])
                with sign_in_tab:
                    st.caption("See your saved assessments and track your risk over time.")
                    _sign_in_form("patient")
                with register_tab:
                    st.caption(
                        "New here? Register with your email. We'll send a link to confirm it."
                    )
                    _register_form()
            with admin_tab:
                st.caption(
                    "For clinic staff. Admins can view every patient's records. Accounts are "
                    "issued by the system owner."
                )
                _sign_in_form("admin")
        st.caption(
            "&#128274; Patient passwords are kept by Supabase Auth and admin passwords only "
            f"as salted hashes. Sessions end after {auth.SESSION_HOURS} hours or when you "
            "sign out."
        )


st.set_page_config(page_title="CardioRisk · Heart Health", page_icon="❤️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

try:
    bundle = load_bundle(ARTIFACT)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.code("uv run python -m src.train\nuv run streamlit run app.py")
    st.stop()

metrics = bundle.get("metrics", {})
account = st.session_state.get("account")
is_admin = bool(account) and account["role"] == "admin"
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
if is_admin:
    # Admins only need to hear about problems, not the service details.
    pills = [pill for pill in pills if pill[0] == "warn"]
if account:
    who = account["username"] if is_admin else (account.get("name") or account["username"])
    pills.append(("ok", f"{'Admin' if is_admin else 'Patient'} · {who}"))
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

if not account:
    _render_login()
    st.stop()

with st.sidebar:
    st.markdown("### Admin" if is_admin else "### My account")
    if is_admin:
        st.caption(f"Signed in as **{account['username']}**. All patient records are visible.")
    else:
        st.caption(
            f"Signed in as **{account.get('name') or account['username']}**  \n"
            f"{account.get('email') or account['username']}  \n"
            f"Patient ID `{account['patient_id']}`"
        )
    if st.button("Sign out", width="stretch"):
        _sign_out()
        st.session_state["signed_out"] = True
        st.rerun()
    st.divider()
    if not is_admin:
        st.markdown("### Model")
        left, right = st.columns(2)
        left.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}" if "roc_auc" in metrics else "n/a")
        right.metric("Accuracy", _pct(metrics.get("accuracy")))
        left.metric("Recall", _pct(metrics.get("recall")))
        right.metric("Precision", _pct(metrics.get("precision")))
        st.caption(
            f"Version {metrics.get('model_version', 'n/a')} · "
            f"{metrics.get('dataset_rows', 'n/a')} training patients · CV accuracy "
            f"{_pct(metrics.get('cv_accuracy'))} ± {_pct(metrics.get('cv_accuracy_std'))}"
        )
        st.caption(f"Data hash `{metrics.get('data_hash', 'n/a')}`")
        st.divider()
        st.markdown("### About")
        st.caption(
            "Logistic-regression model trained on the UCI Cleveland heart-disease study. It "
            "estimates the probability of angiographic coronary artery disease from 13 "
            "routine clinical measurements."
        )
    st.warning(
        "Decision support only — not a diagnosis. Always confirm with a qualified clinician."
    )

if is_admin:
    tab_single, tab_records = st.tabs(["New assessment", "Patient records"])
else:
    tab_single, tab_mine = st.tabs(["New assessment", "My history"])

with tab_single:
    form_column, result_column = st.columns([3, 2], gap="large")
    with form_column:
        with st.form("assessment", border=False):
            values: dict[str, float] = {}
            with st.container(border=True, key="card_patient"):
                if is_admin:
                    _section("Patient", "Optional — link this assessment to a patient record.")
                    left, right = st.columns(2)
                    patient_id = left.text_input("Patient ID", placeholder="e.g. P-1001")
                    patient_name = right.text_input("Full name", placeholder="e.g. Asha Rao")
                    patient_note = st.text_input(
                        "Clinical notes",
                        placeholder="e.g. referred for chest discomfort on exertion",
                    )
                else:
                    _section(
                        "Patient",
                        "This assessment will be saved to your own record.",
                    )
                    st.markdown(
                        f"**{html.escape(account.get('name') or account['username'])}** · "
                        f"Patient ID `{account['patient_id']}`"
                    )
                    patient_id = patient_name = patient_note = ""
                test_date = st.date_input(
                    "Test date",
                    value=date.today(),
                    max_value=date.today(),
                    format="DD/MM/YYYY",
                    help="When these measurements were taken. Defaults to today.",
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
        if is_admin:
            identity = {
                "patient_id": patient_id.strip() or None,
                "name": patient_name.strip() or None,
                "notes": patient_note.strip() or None,
            }
        else:
            identity = {"patient_id": account["patient_id"], "name": None, "notes": None}
        ordered = {feature: values[feature] for feature in FEATURES}
        with st.spinner("Scoring and saving…"):
            outcome = _assess(ordered, identity, metrics, account["token"], test_date)
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

if not is_admin:
    with tab_mine:
        record, note, expired = _signed_in_read(
            "/me/history",
            lambda: database.get_patient_with_history(account["patient_id"]),
            account["token"],
            "patient",
        )
        if expired:
            _expire_session()
        if record:
            _render_record(record, note, own=True)
        else:
            st.info(f"Could not load your history: {note}")

else:
    token = account["token"]
    with tab_records:
        last_patient = st.session_state.get("last_patient_id", "")
        search_column, _ = st.columns([2, 3])
        typed = search_column.text_input(
            "Find a patient", placeholder=last_patient or "Patient ID, e.g. P-1001",
            key="history_lookup",
        ).strip()
        lookup_id = typed or last_patient
        if lookup_id:
            record, note, expired = _signed_in_read(
                f"/patients/{quote(lookup_id, safe='')}/history",
                lambda: database.get_patient_with_history(lookup_id),
                token,
                "admin",
            )
            if expired:
                _expire_session()
            if record:
                _render_record(record, note, own=False)
            elif "404" in note:
                st.info(f"No patient found with ID {lookup_id!r}.")
            else:
                st.info(f"Could not load history for {lookup_id!r}: {note}")
        else:
            st.caption("Search by Patient ID to open a patient's record.")

        with st.expander("All patients", expanded=not lookup_id):
            patients_data, patients_note, expired = _signed_in_read(
                "/patients",
                lambda: {"patients": database.list_patients()},
                token,
                "admin",
            )
            if expired:
                _expire_session()
            if patients_note and patients_data is not None:
                st.caption(patients_note)
            if patients_data is not None and patients_data["patients"]:
                directory = pd.DataFrame(patients_data["patients"])
                directory["created_at"] = _local_times(directory["created_at"])
                directory["last_assessed_on"] = pd.to_datetime(directory["last_assessed_on"])
                directory["name"] = directory["name"].fillna("")
                st.dataframe(
                    directory[
                        ["patient_id", "name", "prediction_count", "latest_risk",
                         "last_assessed_on", "created_at"]
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "patient_id": "Patient ID",
                        "name": "Name",
                        "prediction_count": st.column_config.NumberColumn("Assessments"),
                        "latest_risk": st.column_config.ProgressColumn(
                            "Latest risk", min_value=0.0, max_value=1.0, format="percent"
                        ),
                        "last_assessed_on": st.column_config.DateColumn(
                            "Last test", format="D MMM YYYY"
                        ),
                        "created_at": st.column_config.DateColumn(
                            "Registered", format="D MMM YYYY"
                        ),
                    },
                )
            elif patients_data is not None:
                st.info("No patients stored yet.")
            else:
                st.info(f"Could not list patients: {patients_note}")

st.markdown(
    '<div class="footer">CardioRisk is an educational MLOps project. Estimates come from a '
    "statistical model and are not a medical diagnosis.</div>",
    unsafe_allow_html=True,
)
