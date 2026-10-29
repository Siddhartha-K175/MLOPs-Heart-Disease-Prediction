"""Admin and patient accounts with sign-in sessions.

Admins read every patient record; a patient account is linked to one patient
record and sees only that history. Passwords are stored as salted scrypt
hashes; a sign-in returns a random bearer token whose SHA-256 is kept in
``sessions`` and expires after ``SESSION_HOURS``. Usernames are case-insensitive.

Patients register on the website. Manage accounts from a terminal (passwords
are prompted, never echoed)::

    uv run python -m src.auth create-admin admin
    uv run python -m src.auth create-patient-login asha --patient-id P-1001
    uv run python -m src.auth list-accounts
    uv run python -m src.auth delete-account asha
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import re
import secrets
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from src import database

SESSION_HOURS = 8
MIN_PASSWORD_LENGTH = 10
ROLES = ("admin", "patient")
_USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
# Verified against when the username is unknown, so a miss costs the same time
# as a wrong password and does not reveal which usernames exist.
_DUMMY_HASH: str | None = None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    encode = base64.b64encode
    return (
        f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}$"
        f"{encode(salt).decode()}${encode(digest).decode()}"
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def normalize_username(username: str) -> str:
    """Lower-case and validate: 3-64 letters, digits, dots, dashes or underscores."""
    cleaned = username.strip().lower()
    if not _USERNAME.match(cleaned):
        raise ValueError(
            "Username must be 3-64 characters: letters, digits, '.', '-' or '_'."
        )
    return cleaned


def _check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")


def create_admin(username: str, password: str, schema: str | None = None) -> None:
    """Create an admin, or reset an existing admin's password."""
    username = normalize_username(username)
    _check_password(password)
    existing = database.get_account(username, schema=schema)
    if existing and existing["role"] != "admin":
        raise ValueError(f"{username!r} is a patient account, not an admin.")
    database.save_account(username, hash_password(password), "admin", schema=schema)


def create_patient_login(
    username: str, password: str, patient_id: str, schema: str | None = None
) -> None:
    """Give an existing patient record a sign-in (used by staff, not self-service)."""
    username = normalize_username(username)
    _check_password(password)
    if database.get_patient(patient_id, schema=schema) is None:
        raise ValueError(f"No patient with id {patient_id!r}.")
    existing = database.get_account(username, schema=schema)
    if existing and existing["patient_id"] != patient_id:
        raise ValueError("That username is already taken.")
    database.save_account(
        username, hash_password(password), "patient", patient_id=patient_id, schema=schema
    )


def register_patient(
    username: str, password: str, name: str, schema: str | None = None
) -> str:
    """Self-service patient sign-up: creates a new patient record and its account."""
    username = normalize_username(username)
    _check_password(password)
    name = name.strip()
    if not name or len(name) > 120:
        raise ValueError("Enter your full name (up to 120 characters).")
    return database.register_patient(username, hash_password(password), name, schema=schema)


def login(username: str, password: str, schema: str | None = None) -> dict[str, Any] | None:
    """Check credentials; on success start a session and return its token and role."""
    global _DUMMY_HASH
    try:
        username = normalize_username(username)
        account = database.get_account(username, schema=schema)
    except ValueError:
        account = None
    if account is None:
        _DUMMY_HASH = _DUMMY_HASH or hash_password(secrets.token_hex(8))
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, account["password_hash"]):
        return None
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + timedelta(hours=SESSION_HOURS)
    database.create_session(_token_hash(token), username, expires_at, schema=schema)
    return {
        "access_token": token,
        "username": username,
        "role": account["role"],
        "patient_id": account["patient_id"],
        "expires_at": expires_at,
    }


def current_session(token: str | None, schema: str | None = None) -> dict[str, Any] | None:
    """``{username, role, patient_id, name, expires_at}`` for a live token, else ``None``."""
    if not token:
        return None
    return database.get_session(_token_hash(token), schema=schema)


def logout(token: str, schema: str | None = None) -> None:
    database.delete_session(_token_hash(token), schema=schema)


def _prompt_password(username: str) -> str | None:
    password = getpass.getpass(f"Password for {username}: ")
    if password != getpass.getpass("Repeat password: "):
        print("Passwords do not match.", file=sys.stderr)
        return None
    return password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage CardioRisk sign-in accounts.")
    commands = parser.add_subparsers(dest="command", required=True)
    admin = commands.add_parser("create-admin", help="Create an admin or reset its password")
    admin.add_argument("username")
    patient = commands.add_parser(
        "create-patient-login", help="Give an existing patient record a sign-in"
    )
    patient.add_argument("username")
    patient.add_argument("--patient-id", required=True)
    commands.add_parser("list-accounts", help="List admin and patient accounts")
    remove = commands.add_parser("delete-account", help="Delete a sign-in account")
    remove.add_argument("username")
    args = parser.parse_args(argv)

    try:
        database.init_db()
        if args.command == "create-admin":
            password = _prompt_password(args.username)
            if password is None:
                return 1
            create_admin(args.username, password)
            print(f"Admin {args.username.lower()!r} saved. Existing sessions were signed out.")
        elif args.command == "create-patient-login":
            password = _prompt_password(args.username)
            if password is None:
                return 1
            create_patient_login(args.username, password, args.patient_id)
            print(f"Patient login {args.username.lower()!r} linked to {args.patient_id}.")
        elif args.command == "list-accounts":
            accounts = database.list_accounts()
            if not accounts:
                print("No accounts yet.")
            for account in accounts:
                link = f" -> {account['patient_id']}" if account["patient_id"] else ""
                last = account["last_login_at"] or "never"
                print(f"{account['role']:<8} {account['username']}{link}  "
                      f"(created {account['created_at']:%Y-%m-%d}, last sign-in {last})")
        elif database.delete_account(normalize_username(args.username)):
            print(f"Account {args.username.lower()!r} deleted (medical records are kept).")
        else:
            print(f"No account named {args.username!r}.", file=sys.stderr)
            return 1
    except (database.DatabaseError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
