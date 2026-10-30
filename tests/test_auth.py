"""Tests for admin and patient accounts and sessions (src/auth.py)."""

from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from src import auth, database

PASSWORD = "correct horse battery"


def test_password_hash_round_trip_and_salting():
    first = auth.hash_password(PASSWORD)
    second = auth.hash_password(PASSWORD)
    assert first != second  # random salt per hash
    assert PASSWORD not in first
    assert auth.verify_password(PASSWORD, first)
    assert not auth.verify_password("wrong password", first)


def test_verify_rejects_malformed_hashes():
    assert not auth.verify_password(PASSWORD, "")
    assert not auth.verify_password(PASSWORD, "plain-text-password")
    assert not auth.verify_password(PASSWORD, "md5$1$1$1$abc$def")


@pytest.mark.parametrize("bad", ["ab", "has space", "x" * 65, "-leading", "a@b.com"])
def test_admin_usernames_are_validated(bad):
    with pytest.raises(ValueError, match="Username"):
        auth.normalize_username(bad)


@pytest.mark.parametrize("bad", ["", "plainaddress", "a@b", "two@@at.com", "sp ace@x.com"])
def test_emails_are_validated(bad):
    with pytest.raises(ValueError, match="valid email"):
        auth.normalize_email(bad)


def test_usernames_are_case_insensitive(db_schema):
    auth.create_admin("Admin", PASSWORD)
    assert auth.login("ADMIN", PASSWORD)["username"] == "admin"


def test_create_admin_enforces_password_length(db_schema):
    with pytest.raises(ValueError, match="at least"):
        auth.create_admin("admin", "short")


def test_admin_login_issues_a_working_token(db_schema):
    auth.create_admin("admin", PASSWORD)
    session = auth.login("admin", PASSWORD)
    assert session["role"] == "admin"
    assert session["patient_id"] is None
    current = auth.current_session(session["access_token"])
    assert current["username"] == "admin"
    assert current["role"] == "admin"


def test_register_sends_confirmation_without_creating_a_record(db_schema, fake_supabase):
    email = auth.register_patient(" Asha@Example.com ", PASSWORD, "Asha Rao")
    assert email == "asha@example.com"
    assert fake_supabase.sent == [("confirm", "asha@example.com")]
    assert database.list_patients() == []  # nothing until the patient confirms and signs in


def test_unconfirmed_patient_cannot_sign_in(db_schema, fake_supabase):
    auth.register_patient("asha@example.com", PASSWORD, "Asha Rao")
    with pytest.raises(auth.SignInError, match="confirm your email"):
        auth.login("asha@example.com", PASSWORD)


def test_first_confirmed_sign_in_creates_patient_record(db_schema, fake_supabase):
    auth.register_patient("asha@example.com", PASSWORD, "Asha Rao")
    fake_supabase.confirm("asha@example.com")
    session = auth.login("ASHA@example.com", PASSWORD)
    assert session["role"] == "patient"
    assert session["email"] == "asha@example.com"
    assert session["patient_id"].startswith("P-")
    assert database.get_patient(session["patient_id"])["name"] == "Asha Rao"
    current = auth.current_session(session["access_token"])
    assert current["name"] == "Asha Rao"
    assert current["email"] == "asha@example.com"

    again = auth.login("asha@example.com", PASSWORD)
    assert again["patient_id"] == session["patient_id"]  # same record on later sign-ins
    assert len(database.list_patients()) == 1


def test_patient_wrong_password_returns_none(db_schema, fake_supabase):
    auth.register_patient("asha@example.com", PASSWORD, "Asha Rao")
    fake_supabase.confirm("asha@example.com")
    assert auth.login("asha@example.com", "wrong password!") is None
    assert auth.login("nobody@example.com", PASSWORD) is None


def test_register_validates_input(db_schema, fake_supabase):
    with pytest.raises(ValueError, match="full name"):
        auth.register_patient("asha@example.com", PASSWORD, "   ")
    with pytest.raises(ValueError, match="at least"):
        auth.register_patient("asha@example.com", "short", "Asha Rao")
    with pytest.raises(ValueError, match="valid email"):
        auth.register_patient("not-an-email", PASSWORD, "Asha Rao")
    auth.register_patient("asha@example.com", PASSWORD, "Asha Rao")
    with pytest.raises(ValueError, match="already exists"):
        auth.register_patient("asha@example.com", PASSWORD, "Asha Rao")


def test_only_the_token_hash_is_stored(db_schema):
    auth.create_admin("admin", PASSWORD)
    token = auth.login("admin", PASSWORD)["access_token"]
    with closing(database.connect()) as connection:
        stored = connection.execute("SELECT token_hash FROM sessions").fetchone()
    assert stored["token_hash"] != token
    assert len(stored["token_hash"]) == 64


def test_login_rejects_bad_credentials(db_schema):
    auth.create_admin("admin", PASSWORD)
    assert auth.login("admin", "wrong password") is None
    assert auth.login("nobody", PASSWORD) is None
    assert auth.login("bad name!", PASSWORD) is None


def test_invalid_and_expired_tokens_are_rejected(db_schema):
    auth.create_admin("admin", PASSWORD)
    assert auth.current_session(None) is None
    assert auth.current_session("made-up-token") is None
    database.create_session(
        auth._token_hash("old-token"), "admin", datetime.now(UTC) - timedelta(minutes=1)
    )
    assert auth.current_session("old-token") is None


def test_logout_and_password_reset_end_sessions(db_schema):
    auth.create_admin("admin", PASSWORD)
    first = auth.login("admin", PASSWORD)["access_token"]
    auth.logout(first)
    assert auth.current_session(first) is None

    second = auth.login("admin", PASSWORD)["access_token"]
    auth.create_admin("admin", "a brand new password")
    assert auth.current_session(second) is None
    assert auth.login("admin", PASSWORD) is None
    assert auth.login("admin", "a brand new password") is not None


def test_old_admin_tables_are_migrated(db_schema):
    with closing(database.connect()) as connection:
        connection.execute("DROP TABLE sessions")
        connection.execute("DELETE FROM accounts")
        connection.execute(
            "CREATE TABLE admins (username VARCHAR(64) PRIMARY KEY, password_hash TEXT, "
            "created_at TIMESTAMPTZ DEFAULT now(), last_login_at TIMESTAMPTZ)"
        )
        connection.execute("CREATE TABLE admin_sessions (token_hash CHAR(64))")
        connection.execute(
            "INSERT INTO admins (username, password_hash) VALUES (%s, %s)",
            ("Legacy", auth.hash_password(PASSWORD)),
        )
        connection.commit()
    database.init_db()
    assert auth.login("legacy", PASSWORD)["role"] == "admin"
    with closing(database.connect()) as connection:
        leftover = connection.execute(
            "SELECT count(*) AS n FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name LIKE 'admin%%'"
        ).fetchone()
    assert leftover["n"] == 0


def test_cli_lists_and_deletes_accounts(db_schema, fake_supabase, capsys):
    auth.create_admin("admin", PASSWORD)
    auth.register_patient("asha@example.com", PASSWORD, "Asha Rao")
    fake_supabase.confirm("asha@example.com")
    patient_id = auth.login("asha@example.com", PASSWORD)["patient_id"]
    assert auth.main(["list-accounts"]) == 0
    out = capsys.readouterr().out
    assert "admin" in out and f"asha@example.com -> {patient_id}" in out
    assert auth.main(["delete-account", "Asha@Example.com"]) == 0
    assert database.get_patient(patient_id) is not None  # medical record kept
    assert auth.main(["delete-account", "asha@example.com"]) == 1
