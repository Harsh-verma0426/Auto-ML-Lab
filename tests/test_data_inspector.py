import pandas as pd
import pytest

from automl.data_inspector import (
    infer_column_type,
    inspect_dataframe,
)


def test_infers_numeric_type():
    series = pd.Series([10, 20, 30])

    assert infer_column_type(series) == "numeric"


def test_detects_missing_values():
    df = pd.DataFrame({
        "age": [20, None, 30],
    })

    report = inspect_dataframe(df)

    assert report["columns"]["age"]["missing_count"] == 1
    assert report["columns"]["age"]["missing_percent"] == 33.33


def test_detects_duplicate_rows():
    df = pd.DataFrame({
        "age": [20, 20, 30],
        "plan": ["basic", "basic", "premium"],
    })

    report = inspect_dataframe(df)

    assert report["duplicate_rows"] == 1


def test_detects_constant_columns():
    df = pd.DataFrame({
        "region": ["north", "north", "north"],
    })

    report = inspect_dataframe(df)

    assert report["columns"]["region"]["is_constant"] is True


def test_detects_all_missing_columns():
    df = pd.DataFrame({
        "value": [None, None, None],
    })

    report = inspect_dataframe(df)

    assert report["columns"]["value"]["all_missing"] is True
    assert report["columns"]["value"]["is_constant"] is False


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        inspect_dataframe(pd.DataFrame())