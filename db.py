"""Run ad-hoc SQL against the Supabase application database from the command line.

Useful when you are not in the Supabase SQL Editor. Examples:

    uv run python db.py "SELECT patient_id, name FROM patients;"
    uv run python db.py "SELECT * FROM patient_history ORDER BY created_at DESC LIMIT 5;"
    uv run python db.py "UPDATE patients SET notes = 'reviewed' WHERE patient_id = 'P-1001';"

One statement per run. Connects with SUPABASE_DB_URL (see src/database.py and
.env) with search_path set to DB_SCHEMA; pick another schema with
``--schema NAME``. A missing schema is created and initialized from
``schema.sql`` automatically.
"""

from __future__ import annotations

import argparse
import sys

import psycopg

from src import database


def run_sql(
    connection: psycopg.Connection, sql: str
) -> tuple[list[str] | None, list[tuple], int]:
    """Execute one statement. Returns (columns, rows, affected_rowcount)."""
    with connection.cursor() as cursor:
        cursor.execute(sql)
        if cursor.description is None:
            connection.commit()
            return None, [], cursor.rowcount
        columns = [column.name for column in cursor.description]
        rows = [tuple(row[column] for column in columns) for row in cursor.fetchall()]
    return columns, rows, len(rows)


def _render(columns: list[str], rows: list[tuple]) -> str:
    """Format rows as an aligned text table with NULL for missing values."""
    cells = [["NULL" if value is None else str(value) for value in row] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in cells)) if cells else len(column)
        for index, column in enumerate(columns)
    ]
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    rule = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in cells
    ]
    return "\n".join([header, rule, *body])


def _connect(target: str) -> psycopg.Connection:
    """Connect, creating and initializing the schema when it does not exist yet."""
    if not database.schema_exists(target):
        database.init_db(target)
        print(f"Created and initialized schema `{target}` from schema.sql")
    return database.connect(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one SQL statement against the Supabase patients database.",
        epilog='Example: uv run python db.py "SELECT * FROM patient_history LIMIT 5;"',
    )
    parser.add_argument("sql", nargs="?", help="SQL statement (reads stdin when omitted)")
    parser.add_argument(
        "--schema",
        default=database.DB_SCHEMA,
        help="Postgres schema to open (default: %(default)s or DB_SCHEMA)",
    )
    args = parser.parse_args(argv)

    sql = (args.sql or sys.stdin.read()).strip()
    if not sql:
        parser.error("no SQL statement given")

    try:
        connection = _connect(args.schema)
    except (database.DatabaseError, ValueError) as error:
        print(f"Connection error: {error}", file=sys.stderr)
        print("Check SUPABASE_DB_URL in .env.", file=sys.stderr)
        return 1

    try:
        columns, rows, affected = run_sql(connection, sql)
    except database.DatabaseError as error:
        print(f"SQL error: {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()

    if columns is None:
        print(f"{affected} row(s) affected.")
    else:
        print(_render(columns, rows))
        print(f"({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
