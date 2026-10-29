"""Exploratory data analysis for the heart-disease dataset.

Produces the standard chart set from the workflow (target balance, missingness,
correlation, box plots, scatter trends, group-by) as PNGs plus a text report.

Usage:
    python -m src.eda
    python -m src.eda --data data/heart.csv --output artifacts/eda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .model import FEATURES
from .profile import missing_report
from .train import DEFAULT_URL, TARGET, fetch_raw, validate_frame

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA = Path("data/heart.csv")
DEFAULT_OUTPUT = Path("artifacts/eda")
CHARTS = (
    "target_distribution.png",
    "missing_values.png",
    "correlation_heatmap.png",
    "boxplots.png",
    "scatter_trends.png",
    "groupby_target.png",
)


def outlier_report(frame: pd.DataFrame) -> dict[str, int]:
    """Count rows outside the 1.5*IQR fences per feature."""
    counts: dict[str, int] = {}
    for feature in FEATURES:
        values = frame[feature]
        first, third = values.quantile(0.25), values.quantile(0.75)
        spread = third - first
        lower, upper = first - 1.5 * spread, third + 1.5 * spread
        outside = int(((values < lower) | (values > upper)).sum())
        if outside:
            counts[feature] = outside
    return counts


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_target_distribution(frame: pd.DataFrame, path: Path) -> None:
    """Class balance, the first thing the video checks."""
    counts = frame[TARGET].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = [f"class {index}" for index in counts.index]
    bars = axis.bar(labels, counts.values, color=["#4c72b0", "#c44e52"])
    axis.bar_label(bars, labels=[f"{value} ({value / len(frame):.1%})" for value in counts.values])
    axis.set_title("Target distribution")
    axis.set_ylabel("Rows")
    _save(figure, path)


def plot_missing_values(missing: pd.DataFrame, path: Path) -> None:
    """Missing-value percentages, before any cleaning."""
    figure, axis = plt.subplots(figsize=(10, 4))
    colors = ["#c44e52" if value > 0 else "#55a868" for value in missing["percent"]]
    axis.bar(missing.index, missing["percent"], color=colors)
    axis.set_title("Missing values by column (raw data)")
    axis.set_ylabel("% missing")
    axis.tick_params(axis="x", rotation=45)
    if missing["missing"].sum() == 0:
        axis.text(0.5, 0.9, "no missing values", transform=axis.transAxes, ha="center")
    _save(figure, path)


def plot_correlation_heatmap(frame: pd.DataFrame, path: Path) -> None:
    """Feature correlation heatmap."""
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        frame.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=axis,
        cbar_kws={"shrink": 0.8},
    )
    axis.set_title("Correlation heatmap")
    _save(figure, path)


def plot_boxplots(frame: pd.DataFrame, path: Path) -> None:
    """Box plots per feature to spot outliers (e.g. cholesterol)."""
    columns = len(FEATURES)
    rows = (columns + 3) // 4
    figure, axes = plt.subplots(rows, 4, figsize=(14, 3 * rows))
    for axis, feature in zip(axes.flat, FEATURES, strict=False):
        sns.boxplot(x=frame[feature], ax=axis, color="#4c72b0")
        axis.set_title(feature, fontsize=9)
    for axis in list(axes.flat)[columns:]:
        axis.set_visible(False)
    figure.suptitle("Box plots by feature", y=1.01)
    _save(figure, path)


def plot_scatter_trends(frame: pd.DataFrame, path: Path) -> None:
    """Scatter with regression trend per class (age vs max heart rate)."""
    grid = sns.lmplot(
        data=frame,
        x="age",
        y="thalach",
        hue=TARGET,
        height=5,
        aspect=1.2,
        scatter_kws={"alpha": 0.5},
    )
    grid.figure.suptitle("Age vs maximum heart rate by class", y=1.02)
    grid.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(grid.figure)


def plot_groupby_target(frame: pd.DataFrame, path: Path) -> None:
    """Standardised feature means per class (group-by comparison)."""
    grouped = frame.groupby(TARGET)[FEATURES].mean()
    spread = grouped.std().replace(0, 1)
    standardised = (grouped - grouped.mean()) / spread
    figure, axis = plt.subplots(figsize=(10, 6))
    standardised.T.plot(kind="barh", ax=axis, color=["#4c72b0", "#c44e52"])
    axis.set_title("Feature means by class (standardised)")
    axis.set_xlabel("Standardised mean")
    axis.legend(title=TARGET)
    _save(figure, path)


def run_eda(source: str | Path | None, output: Path) -> dict:
    """Generate every chart and the text report; returns the summary."""
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid")
    output.mkdir(parents=True, exist_ok=True)

    raw = fetch_raw(source)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    frame = validate_frame(raw.copy())
    missing = missing_report(raw)
    outliers = outlier_report(frame)
    duplicates = int(raw.duplicated().sum())
    target_counts = frame[TARGET].value_counts().sort_index()

    missing.to_csv(output / "missing_values.csv")
    plot_target_distribution(frame, output / "target_distribution.png")
    plot_missing_values(missing, output / "missing_values.png")
    plot_correlation_heatmap(frame, output / "correlation_heatmap.png")
    plot_boxplots(frame, output / "boxplots.png")
    plot_scatter_trends(frame, output / "scatter_trends.png")
    plot_groupby_target(frame, output / "groupby_target.png")

    summary = {
        "source": str(source or DEFAULT_URL),
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(frame)),
        "missing_rows": int(missing["missing"].gt(0).sum()),
        "duplicate_rows": duplicates,
        "target_counts": {int(key): int(value) for key, value in target_counts.items()},
        "outliers": outliers,
        "charts": [str(output / name) for name in CHARTS],
    }
    lines = [
        "Heart-disease EDA report",
        f"source: {summary['source']}",
        f"rows: {summary['rows_raw']} raw -> {summary['rows_clean']} cleaned",
        f"columns with missing values: {summary['missing_rows']}",
        f"duplicate rows: {duplicates}",
        "target counts: " + ", ".join(f"{k}: {v}" for k, v in summary["target_counts"].items()),
        "IQR outliers: "
        + (", ".join(f"{k}: {v}" for k, v in outliers.items()) if outliers else "none"),
        "charts: " + ", ".join(CHARTS),
    ]
    report = "\n".join(lines) + "\n"
    (output / "report.txt").write_text(report, encoding="utf-8")
    for line in lines:
        LOGGER.info("%s", line)
    return summary


def create_parser() -> argparse.ArgumentParser:
    """Create the EDA CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="Raw CSV to explore.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    """Run EDA and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = create_parser().parse_args()
    source = args.data
    if source is None and DEFAULT_DATA.exists():
        source = DEFAULT_DATA
    try:
        run_eda(source, args.output)
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    LOGGER.info("EDA written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
