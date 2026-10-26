
import json

import pandas as pd
import pytest

from automl.eda import generate_eda


def test_generates_report_and_plots(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "spend": [10.0, 20.0, 15.0, 25.0],
        "plan": ["basic", "premium", "basic", "premium"],
        "churn": [0, 1, 0, 1],
    })

    report = generate_eda(
        df,
        output_dir=tmp_path,
        target="churn",
    )

    assert report["shape"] == {"rows": 4, "columns": 4}
    assert "age" in report["numeric_summary"]
    assert "plan" in report["categorical_summary"]
    assert "correlation_heatmap.png" in report["plots"]
    assert "target_distribution.png" in report["plots"]
    assert (tmp_path / "eda_report.json").exists()
    assert len(report["plots"]) > 0


def test_missing_values_are_reported(tmp_path):
    df = pd.DataFrame({
        "age": [20, None, 40],
        "spend": [10.0, 15.0, 20.0],
    })

    report = generate_eda(df, output_dir=tmp_path)

    assert report["missing_values"]["count_by_column"]["age"] == 1
    assert report["missing_values"]["percent_by_column"]["age"] == pytest.approx(
        33.33, abs=0.01
    )


def test_unknown_target_is_rejected(tmp_path):
    df = pd.DataFrame({"age": [20, 30]})

    with pytest.raises(ValueError, match="Target column not found"):
        generate_eda(df, output_dir=tmp_path, target="churn")


def test_empty_dataframe_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty dataset"):
        generate_eda(pd.DataFrame(), output_dir=tmp_path)


def test_report_is_valid_json(tmp_path):
    df = pd.DataFrame({
        "age": [20, None, 40],
        "score": [0.5, 0.7, 0.9],
    })

    generate_eda(df, output_dir=tmp_path)

    with (tmp_path / "eda_report.json").open(
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    assert report["shape"]["rows"] == 3
    assert report["missing_values"]["total_cells_missing"] == 1
