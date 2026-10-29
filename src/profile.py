"""Data-quality profiling: missingness percentages and duplicate rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

PROFILE_KEYS = ("rows", "columns", "rows_with_missing", "missing_cells", "duplicate_rows")


def missing_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages for the raw frame."""
    counts = raw.isna().sum()
    return pd.DataFrame(
        {
            "missing": counts,
            "percent": (counts / max(len(raw), 1) * 100).round(3),
        }
    )


def data_profile(raw: pd.DataFrame) -> dict[str, Any]:
    """Absolute counts of rows, missing cells, and duplicates in a raw frame."""
    missing = raw.isna().sum()
    return {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "rows_with_missing": int(raw.isna().any(axis=1).sum()),
        "missing_cells": int(missing.sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "missing_by_column": {
            column: int(count) for column, count in missing.items() if count > 0
        },
    }


def combine_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge profiles of disjoint frames by summing their absolute counts."""
    combined = {key: 0 for key in PROFILE_KEYS if key != "columns"}
    combined["columns"] = max((int(profile["columns"]) for profile in profiles), default=0)
    for profile in profiles:
        for key in PROFILE_KEYS:
            if key == "columns":
                continue
            combined[key] += int(profile[key])
        for column, count in profile["missing_by_column"].items():
            combined.setdefault("missing_by_column", {})
            combined["missing_by_column"][column] = (
                combined["missing_by_column"].get(column, 0) + int(count)
            )
    combined.setdefault("missing_by_column", {})
    return combined


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Add missingness percentages to an absolute-count profile."""
    rows = max(int(profile["rows"]), 1)
    columns = max(int(profile["columns"]), 1)
    return {
        **profile,
        "missing_cells_pct": round(profile["missing_cells"] / (rows * columns) * 100, 3),
        "rows_with_missing_pct": round(profile["rows_with_missing"] / rows * 100, 3),
        "duplicate_rows_pct": round(profile["duplicate_rows"] / rows * 100, 3),
        "missing_percent_by_column": {
            column: round(count / rows * 100, 3)
            for column, count in profile["missing_by_column"].items()
        },
    }
