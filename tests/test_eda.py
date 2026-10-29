import pandas as pd

from src.eda import CHARTS, outlier_report, run_eda


def test_outlier_report_flags_extreme_values(heart_frame):
    frame = heart_frame.copy()
    frame.loc[frame.index[0], "chol"] = 600.0

    counts = outlier_report(frame)

    assert "chol" in counts
    assert counts["chol"] >= 1


def test_run_eda_writes_charts_and_report(tmp_path, heart_frame):
    source = tmp_path / "input.csv"
    frame = pd.concat([heart_frame, heart_frame.head(1)], ignore_index=True)
    frame.to_csv(source, index=False)
    output = tmp_path / "eda"

    summary = run_eda(source, output)

    for name in CHARTS:
        chart = output / name
        assert chart.exists(), name
        assert chart.stat().st_size > 0, name
    assert (output / "report.txt").exists()
    assert (output / "missing_values.csv").exists()

    assert summary["rows_raw"] == len(frame)
    assert summary["duplicate_rows"] == 1
    assert summary["missing_rows"] == 0
    assert set(summary["target_counts"]) == {0, 1}

    report = (output / "report.txt").read_text(encoding="utf-8")
    assert "duplicate rows: 1" in report
