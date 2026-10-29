"""Supabase (PostgreSQL) persistence for patient records and prediction history.

The SQL lives in ``schema.sql`` at the project root (Postgres dialect). The
tables sit in a dedicated Postgres schema (``DB_SCHEMA``, default
``heart_health``) instead of ``public``, so Supabase's auto-generated REST API
never exposes these health records. The API creates the schema and runs the
file automatically at startup (every statement is idempotent).

Connection settings come from environment variables, optionally stored in a
gitignored ``.env`` file next to this package::

    SUPABASE_DB_URL  postgresql://postgres.<ref>:<password>@<host>:5432/postgres
                     (Supabase dashboard -> Connect -> Session pooler)
    DB_SCHEMA        heart_health
    DB_TEST_SCHEMA   heart_health_test   (used by the test suite)

Each helper opens and closes its own connection so the module is safe to call
from FastAPI's thread pool and from the Streamlit app.
"""

from __future__ import annotations

import os
import re
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from src.model import FEATURES

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema.sql"
ENV_PATH = ROOT / ".env"

# Callers catch this instead of importing the driver themselves.
DatabaseError = psycopg.Error


def _load_env() -> None:
    """Read KEY=VALUE lines from .env without overriding real environment variables."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

DB_URL = os.environ.get("SUPABASE_DB_URL", "")
DB_SCHEMA = os.environ.get("DB_SCHEMA", "heart_health")
TEST_SCHEMA = os.environ.get("DB_TEST_SCHEMA", "heart_health_test")

# Schema names are interpolated into DDL, so they must be plain identifiers.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Insert column names are built only from this whitelist (never from input).
_PREDICTION_COLUMNS = (
    "patient_id", *FEATURES, "prediction", "risk_probability", "risk_label", "model_version",
)


def _check_identifier(name: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValueError(f"Unsafe Postgres schema name: {name!r}")
    return name


def server_label() -> str:
    """Host and port of the configured database, without credentials (for logs)."""
    try:
        parts = urlsplit(DB_URL)
        return f"{parts.hostname}:{parts.port or 5432}" if parts.hostname else "(not configured)"
    except ValueError:
        return "(invalid SUPABASE_DB_URL)"


def _open() -> psycopg.Connection:
    if not DB_URL:
        raise ValueError(
            "SUPABASE_DB_URL is not set. Copy the Session pooler connection string from "
            "the Supabase dashboard (Connect button) into .env."
        )
    # prepare_threshold=None keeps this working behind Supabase's transaction pooler.
    return psycopg.connect(
        DB_URL, row_factory=dict_row, connect_timeout=10, prepare_threshold=None
    )


def connect(schema: str | None = None) -> psycopg.Connection:
    """Open a connection returning dictionary rows, with search_path set to ``schema``."""
    target = _check_identifier(schema or DB_SCHEMA)
    connection = _open()
    connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(target)))
    return connection


def schema_exists(schema: str | None = None) -> bool:
    target = _check_identifier(schema or DB_SCHEMA)
    with closing(_open()) as connection:
        row = connection.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (target,)
        ).fetchone()
    return row is not None


def init_db(schema: str | None = None) -> None:
    """Create the schema if missing, then create tables, indexes, and the view."""
    target = _check_identifier(schema or DB_SCHEMA)
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with closing(_open()) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(target))
        )
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(target)))
        # Without parameters psycopg sends the whole file as one multi-statement query.
        connection.execute(ddl)
        connection.commit()


def drop_db(schema: str) -> None:
    """Drop a schema and everything in it (used by the test suite between tests)."""
    target = _check_identifier(schema)
    if target == "public":
        raise ValueError("Refusing to drop the public schema")
    with closing(_open()) as connection:
        connection.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(target))
        )
        connection.commit()


def recreate_db(schema: str | None = None) -> None:
    """Drop (if present) and recreate the schema with clean tables."""
    target = _check_identifier(schema or TEST_SCHEMA)
    drop_db(target)
    init_db(target)


def _ensure_patient(
    connection: psycopg.Connection,
    patient_id: str | None,
    name: str | None,
    notes: str | None,
) -> str:
    """Insert or update the patient row and return its id."""
    resolved = patient_id or f"P-{os.urandom(4).hex().upper()}"
    # COALESCE keeps the stored name/notes when a resave omits them.
    connection.execute(
        """
        INSERT INTO patients (patient_id, name, notes)
        VALUES (%s, %s, %s)
        ON CONFLICT (patient_id) DO UPDATE SET
            name = COALESCE(EXCLUDED.name, patients.name),
            notes = COALESCE(EXCLUDED.notes, patients.notes)
        """,
        (resolved, name, notes),
    )
    return resolved


def save_prediction(
    *,
    features: dict[str, Any],
    prediction: int,
    risk_probability: float,
    risk_label: str,
    model_version: str | None = None,
    patient_id: str | None = None,
    name: str | None = None,
    notes: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """Persist one scored prediction; returns ``record_id`` and resolved ``patient_id``.

    When neither ``patient_id`` nor ``name`` is given the row is stored
    anonymously (``patient_id`` is NULL) but still kept as an audit record.
    """
    missing = [feature for feature in FEATURES if feature not in features]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")

    with closing(connect(schema)) as connection:
        resolved_id = (
            _ensure_patient(connection, patient_id, name, notes)
            if (patient_id or name)
            else None
        )
        placeholders = ", ".join(["%s"] * len(_PREDICTION_COLUMNS))
        values = [
            resolved_id,
            *[float(features[feature]) for feature in FEATURES],
            int(prediction),
            float(risk_probability),
            risk_label,
            model_version,
        ]
        row = connection.execute(
            f"INSERT INTO predictions ({', '.join(_PREDICTION_COLUMNS)}) "
            f"VALUES ({placeholders}) RETURNING id",
            values,
        ).fetchone()
        connection.commit()
    return {"record_id": row["id"], "patient_id": resolved_id}


def list_patients(schema: str | None = None) -> list[dict[str, Any]]:
    """Every patient with prediction counts, latest score time and latest risk."""
    with closing(connect(schema)) as connection:
        rows = connection.execute(
            """
            SELECT
                p.patient_id,
                p.name,
                p.notes,
                p.created_at,
                COUNT(pr.id)       AS prediction_count,
                MAX(pr.created_at) AS last_prediction_at,
                (
                    SELECT latest.risk_probability FROM predictions AS latest
                    WHERE latest.patient_id = p.patient_id
                    ORDER BY latest.created_at DESC, latest.id DESC
                    LIMIT 1
                )                  AS latest_risk
            FROM patients AS p
            LEFT JOIN predictions AS pr ON pr.patient_id = p.patient_id
            GROUP BY p.patient_id, p.name, p.notes, p.created_at
            ORDER BY last_prediction_at DESC NULLS LAST, p.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_patient(patient_id: str, schema: str | None = None) -> dict[str, Any] | None:
    """Return one patient row, or ``None`` when the id is unknown."""
    with closing(connect(schema)) as connection:
        row = connection.execute(
            "SELECT * FROM patients WHERE patient_id = %s", (patient_id,)
        ).fetchone()
    return dict(row) if row else None


def get_history(
    patient_id: str, limit: int = 500, schema: str | None = None
) -> list[dict[str, Any]]:
    """Predictions linked to ``patient_id`` via the ``patient_history`` view, newest first."""
    with closing(connect(schema)) as connection:
        rows = connection.execute(
            """
            SELECT * FROM patient_history
            WHERE patient_id = %s
            ORDER BY created_at DESC, record_id DESC
            LIMIT %s
            """,
            (patient_id, int(limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def get_patient_with_history(
    patient_id: str, limit: int = 500, schema: str | None = None
) -> dict[str, Any] | None:
    """Patient info plus their prediction history, or ``None`` for unknown ids."""
    patient = get_patient(patient_id, schema=schema)
    if patient is None:
        return None
    history = get_history(patient_id, limit=limit, schema=schema)
    return {"patient": patient, "history": history, "count": len(history)}


def recent_predictions(limit: int = 50, schema: str | None = None) -> list[dict[str, Any]]:
    """The most recent predictions, anonymous ones included."""
    with closing(connect(schema)) as connection:
        rows = connection.execute(
            """
            SELECT * FROM predictions
            ORDER BY id DESC
            LIMIT %s
            """,
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
    return [dict(row) for row in rows]


def save_account(
    username: str,
    password_hash: str,
    role: str,
    patient_id: str | None = None,
    schema: str | None = None,
) -> None:
    """Create an account, or replace the password of an existing one."""
    with closing(connect(schema)) as connection:
        connection.execute(
            """
            INSERT INTO accounts (username, password_hash, role, patient_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash
            """,
            (username, password_hash, role, patient_id),
        )
        # A password change signs the account out everywhere.
        connection.execute("DELETE FROM sessions WHERE username = %s", (username,))
        connection.commit()


def register_patient(
    username: str, password_hash: str, name: str, schema: str | None = None
) -> str:
    """Create a new patient record and its sign-in account together; returns the id."""
    patient_id = f"P-{os.urandom(4).hex().upper()}"
    with closing(connect(schema)) as connection:
        try:
            connection.execute(
                "INSERT INTO patients (patient_id, name) VALUES (%s, %s)", (patient_id, name)
            )
            connection.execute(
                """
                INSERT INTO accounts (username, password_hash, role, patient_id)
                VALUES (%s, %s, 'patient', %s)
                """,
                (username, password_hash, patient_id),
            )
        except psycopg.errors.UniqueViolation as error:
            raise ValueError("That username is already taken.") from error
        connection.commit()
    return patient_id


def get_account(username: str, schema: str | None = None) -> dict[str, Any] | None:
    with closing(connect(schema)) as connection:
        row = connection.execute(
            "SELECT username, password_hash, role, patient_id FROM accounts WHERE username = %s",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def list_accounts(schema: str | None = None) -> list[dict[str, Any]]:
    with closing(connect(schema)) as connection:
        rows = connection.execute(
            """
            SELECT username, role, patient_id, created_at, last_login_at
            FROM accounts ORDER BY role, username
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_account(username: str, schema: str | None = None) -> bool:
    """Delete a sign-in account (the patient's medical record is kept)."""
    with closing(connect(schema)) as connection:
        deleted = connection.execute(
            "DELETE FROM accounts WHERE username = %s", (username,)
        ).rowcount
        connection.commit()
    return deleted > 0


def create_session(
    token_hash: str, username: str, expires_at: Any, schema: str | None = None
) -> None:
    """Store a new session and record the login time (expired sessions are pruned)."""
    with closing(connect(schema)) as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at <= now()")
        connection.execute(
            "INSERT INTO sessions (token_hash, username, expires_at) VALUES (%s, %s, %s)",
            (token_hash, username, expires_at),
        )
        connection.execute(
            "UPDATE accounts SET last_login_at = now() WHERE username = %s", (username,)
        )
        connection.commit()


def get_session(token_hash: str, schema: str | None = None) -> dict[str, Any] | None:
    """The unexpired session for ``token_hash`` with its account's role, or ``None``."""
    with closing(connect(schema)) as connection:
        row = connection.execute(
            """
            SELECT s.username, s.expires_at, a.role, a.patient_id, p.name
            FROM sessions AS s
            JOIN accounts AS a ON a.username = s.username
            LEFT JOIN patients AS p ON p.patient_id = a.patient_id
            WHERE s.token_hash = %s AND s.expires_at > now()
            """,
            (token_hash,),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token_hash: str, schema: str | None = None) -> None:
    with closing(connect(schema)) as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = %s", (token_hash,))
        connection.commit()
