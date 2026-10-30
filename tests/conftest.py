import uuid

import numpy as np
import pandas as pd
import pytest

from src import database


@pytest.fixture
def db_schema(monkeypatch):
    """Point every database helper at a clean, throwaway Postgres schema."""
    if not database.DB_URL:
        pytest.skip("SUPABASE_DB_URL is not set")
    monkeypatch.setattr(database, "DB_SCHEMA", database.TEST_SCHEMA)
    database.recreate_db(database.TEST_SCHEMA)
    yield database.TEST_SCHEMA


@pytest.fixture
def heart_frame() -> pd.DataFrame:
    """Small synthetic frame with the real schema and both target classes."""
    rng = np.random.default_rng(42)
    rows = 120
    frame = pd.DataFrame(
        {
            "age": rng.integers(30, 80, rows),
            "sex": rng.integers(0, 2, rows),
            "cp": rng.integers(1, 5, rows),
            "trestbps": rng.integers(100, 180, rows),
            "chol": rng.integers(150, 400, rows),
            "fbs": rng.integers(0, 2, rows),
            "restecg": rng.integers(0, 3, rows),
            "thalach": rng.integers(90, 200, rows),
            "exang": rng.integers(0, 2, rows),
            "oldpeak": rng.uniform(0, 4, rows).round(1),
            "slope": rng.integers(1, 4, rows),
            "ca": rng.integers(0, 4, rows),
            "thal": rng.integers(3, 8, rows),
        }
    )
    frame["target"] = rng.permutation(np.tile([0, 1], rows // 2))
    return frame


class FakeSupabaseAuth:
    """In-memory stand-in for Supabase Auth so tests never send real emails."""

    def __init__(self):
        self.users: dict[str, dict] = {}
        self.sent: list[tuple[str, str]] = []  # (kind, email)

    def sign_up(self, email, password, full_name):
        from src import supabase_auth

        if email in self.users:
            raise supabase_auth.AuthError(
                "An account with this email already exists. Sign in instead."
            )
        self.users[email] = {
            "id": str(uuid.uuid4()), "email": email, "password": password,
            "confirmed": False, "user_metadata": {"full_name": full_name},
        }
        self.sent.append(("confirm", email))
        return {"id": self.users[email]["id"], "email": email}

    def confirm(self, email):
        self.users[email]["confirmed"] = True

    def sign_in(self, email, password):
        from src import supabase_auth

        user = self.users.get(email)
        if user is None or user["password"] != password:
            raise supabase_auth.AuthError("Incorrect email or password.")
        if not user["confirmed"]:
            raise supabase_auth.AuthError(
                "Please confirm your email first: open the link we sent you."
            )
        return {key: user[key] for key in ("id", "email", "user_metadata")}

    def resend_confirmation(self, email):
        self.sent.append(("confirm", email))


@pytest.fixture
def fake_supabase(monkeypatch):
    from src import supabase_auth

    fake = FakeSupabaseAuth()
    for name in ("sign_up", "sign_in", "resend_confirmation"):
        monkeypatch.setattr(supabase_auth, name, getattr(fake, name))
    return fake
