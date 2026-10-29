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


@pytest.mark.parametrize("bad", ["ab", "has space", "x" * 65, "-leading"])
def test_usernames_are_validated(bad):
    with pytest.raises(ValueError, match="Username"):
        auth.normalize_username(bad)


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


def test_register_patient_creates_record_and_account(db_schema):
    patient_id = auth.register_patient("asha", PASSWORD, "Asha Rao")
    assert patient_id.startswith("P-")
    assert database.get_patient(patient_id)["name"] == "Asha Rao"
    session = auth.login("asha", PASSWORD)
    assert session["role"] == "patient"
    assert session["patient_id"] == patient_id
    assert auth.current_session(session["access_token"])["name"] == "Asha Rao"


def test_register_rejects_taken_username_without_orphan_record(db_schema):
    auth.register_patient("asha", PASSWORD, "Asha Rao")
    with pytest.raises(ValueError, match="taken"):
        auth.register_patient("ASHA", PASSWORD, "Someone Else")
    assert len(database.list_patients()) == 1  # the failed sign-up left nothing behind


def test_register_validates_name_and_password(db_schema):
    with pytest.raises(ValueError, match="full name"):
        auth.register_patient("asha", PASSWORD, "   ")
    with pytest.raises(ValueError, match="at least"):
        auth.register_patient("asha", "short", "Asha Rao")


def test_patient_login_for_existing_record(db_schema):
    database.save_prediction(
        features={"age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 240, "fbs": 0,
                  "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2,
                  "ca": 0, "thal": 3},
        prediction=0, risk_probability=0.2, risk_label="Lower", patient_id="P-1", name="Ravi",
    )
    auth.create_patient_login("ravi", PASSWORD, "P-1")
    assert auth.login("ravi", PASSWORD)["patient_id"] == "P-1"
    with pytest.raises(ValueError, match="No patient"):
        auth.create_patient_login("ghost", PASSWORD, "P-missing")


def test_admin_and_patient_names_cannot_collide(db_schema):
    auth.register_patient("asha", PASSWORD, "Asha Rao")
    with pytest.raises(ValueError, match="patient account"):
        auth.create_admin("asha", PASSWORD)


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


def test_cli_lists_and_deletes_accounts(db_schema, capsys):
    auth.create_admin("admin", PASSWORD)
    patient_id = auth.register_patient("asha", PASSWORD, "Asha Rao")
    assert auth.main(["list-accounts"]) == 0
    out = capsys.readouterr().out
    assert "admin" in out and f"asha -> {patient_id}" in out
    assert auth.main(["delete-account", "asha"]) == 0
    assert database.get_patient(patient_id) is not None  # medical record kept
    assert auth.main(["delete-account", "asha"]) == 1
