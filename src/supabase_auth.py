"""Minimal Supabase Auth (GoTrue) client for patient email sign-up and sign-in.

Supabase emails a confirmation link at sign-up, stores the password and
refuses sign-in until the address is confirmed. The app keeps its own
session tokens (see src/auth.py); this module only proves who the patient is.

Settings (in .env)::

    SUPABASE_URL              https://<project-ref>.supabase.co
    SUPABASE_PUBLISHABLE_KEY  sb_publishable_...  (safe to share; never the secret key)
    APP_URL                   http://localhost:8502  (where the email link returns to)
"""

from __future__ import annotations

from typing import Any

import httpx

from src.database import env  # importing src.database also loads .env

SUPABASE_URL = env("SUPABASE_URL").rstrip("/")
PUBLISHABLE_KEY = env("SUPABASE_PUBLISHABLE_KEY")
APP_URL = env("APP_URL")


class AuthError(Exception):
    """Supabase refused the request; the message is safe to show to the user."""


def configured() -> bool:
    return bool(SUPABASE_URL and PUBLISHABLE_KEY)


def _post(path: str, payload: dict[str, Any], params: dict[str, str] | None = None) -> dict:
    if not configured():
        raise AuthError(
            "Email sign-up is not configured: set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY."
        )
    try:
        response = httpx.post(
            f"{SUPABASE_URL}/auth/v1/{path}",
            params=params,
            json=payload,
            headers={"apikey": PUBLISHABLE_KEY, "Content-Type": "application/json"},
            timeout=15.0,
        )
    except httpx.HTTPError as error:
        raise AuthError(f"Could not reach Supabase Auth ({error.__class__.__name__}).") from error
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code >= 400:
        raise AuthError(_message(body, response.status_code))
    return body


def _message(body: dict, status: int) -> str:
    code = body.get("error_code") or body.get("code") or body.get("error")
    friendly = {
        "email_not_confirmed": "Please confirm your email first: open the link we sent you.",
        "invalid_credentials": "Incorrect email or password.",
        "invalid_grant": "Incorrect email or password.",
        "user_already_exists": "An account with this email already exists. Sign in instead.",
        "weak_password": "That password is too weak. Use at least 10 characters.",
        "over_email_send_rate_limit": (
            "Too many emails were sent recently. Please wait a few minutes and try again."
        ),
        "email_address_invalid": "That email address is not valid.",
        "signup_disabled": "New sign-ups are currently disabled.",
    }
    if code in friendly:
        return friendly[code]
    detail = body.get("msg") or body.get("message") or body.get("error_description")
    return detail or f"Supabase Auth error ({status})."


def sign_up(email: str, password: str, full_name: str) -> dict[str, Any]:
    """Register and trigger the confirmation email. Returns the (unconfirmed) user."""
    params = {"redirect_to": APP_URL} if APP_URL else None
    body = _post(
        "signup",
        {"email": email, "password": password, "data": {"full_name": full_name}},
        params=params,
    )
    return body.get("user") or body


def sign_in(email: str, password: str) -> dict[str, Any]:
    """Check an email and password; returns the confirmed Supabase user."""
    body = _post("token", {"email": email, "password": password},
                 params={"grant_type": "password"})
    user = body.get("user") or {}
    if not user.get("id"):
        raise AuthError("Supabase did not return a user for this sign-in.")
    return user


def resend_confirmation(email: str) -> None:
    params = {"redirect_to": APP_URL} if APP_URL else None
    _post("resend", {"type": "signup", "email": email}, params=params)


def send_password_reset(email: str) -> None:
    params = {"redirect_to": APP_URL} if APP_URL else None
    _post("recover", {"email": email}, params=params)
