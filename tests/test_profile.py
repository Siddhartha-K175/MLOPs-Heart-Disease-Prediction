import pandas as pd

from src.profile import combine_profiles, data_profile, missing_report, summarize_profile


def test_missing_report_counts_and_percent():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})

    report = missing_report(frame)

    assert report.loc["a", "missing"] == 1
    assert report.loc["a", "percent"] == 33.333
    assert report.loc["b", "missing"] == 0


def test_data_profile_counts_missing_and_duplicates():
    frame = pd.DataFrame({"a": [1.0, None, 3.0, 1.0], "b": [1.0, 2.0, 3.0, 1.0]})

    profile = data_profile(frame)

    assert profile["rows"] == 4
    assert profile["columns"] == 2
    assert profile["rows_with_missing"] == 1
    assert profile["missing_cells"] == 1
    assert profile["duplicate_rows"] == 1
    assert profile["missing_by_column"] == {"a": 1}


def test_summarize_profile_adds_percentages():
    frame = pd.DataFrame({"a": [1.0, None, 3.0, 1.0], "b": [1.0, 2.0, 3.0, 1.0]})

    summary = summarize_profile(data_profile(frame))

    assert summary["missing_cells_pct"] == 12.5
    assert summary["rows_with_missing_pct"] == 25.0
    assert summary["duplicate_rows_pct"] == 25.0
    assert summary["missing_percent_by_column"]["a"] == 25.0


def test_combine_profiles_sums_absolute_counts():
    first = pd.DataFrame({"a": [1.0, None], "b": [1.0, 2.0]})
    second = pd.DataFrame({"a": [None, 1.0], "b": [1.0, 2.0]})

    combined = combine_profiles([data_profile(first), data_profile(second)])

    assert combined["rows"] == 4
    assert combined["columns"] == 2
    assert combined["missing_cells"] == 2
    assert combined["missing_by_column"] == {"a": 2}
