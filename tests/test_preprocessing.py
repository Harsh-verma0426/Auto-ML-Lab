
import numpy as np
import pandas as pd
import pytest

from automl.data_quality import diagnose_dataframe
from automl.preprocessing import build_preprocessor


def test_diagnoses_missing_values_and_duplicates():
    df = pd.DataFrame({
        "age": [20, None, 20],
        "plan": ["basic", "premium", "basic"],
    })

    report = diagnose_dataframe(df)

    assert report["duplicate_rows"] == 1
    assert report["columns"]["age"]["missing_count"] == 1
    assert report["columns"]["age"]["missing_percent"] == pytest.approx(
        33.33, abs=0.01
    )


def test_detects_all_missing_and_constant_columns():
    df = pd.DataFrame({
        "empty": [None, None, None],
        "region": ["north", "north", "north"],
    })

    report = diagnose_dataframe(df)

    assert report["columns"]["empty"]["all_missing"] is True
    assert report["columns"]["region"]["constant"] is True


def test_preprocessor_handles_missing_values_and_unseen_categories():
    X_train = pd.DataFrame({
        "age": [20.0, None, 40.0],
        "spend": [10.0, 15.0, None],
        "plan": ["basic", "premium", "basic"],
        "empty": [None, None, None],
    })

    X_test = pd.DataFrame({
        "age": [30.0],
        "spend": [20.0],
        "plan": ["enterprise"],
        "empty": [None],
    })

    preprocessor, metadata = build_preprocessor(X_train)

    # Learn transformation parameters only from training data.
    train_result = preprocessor.fit_transform(X_train)
    test_result = preprocessor.transform(X_test)

    train_array = (
        train_result.toarray()
        if hasattr(train_result, "toarray")
        else np.asarray(train_result)
    )

    test_array = (
        test_result.toarray()
        if hasattr(test_result, "toarray")
        else np.asarray(test_result)
    )

    assert train_array.shape == (3, 4)
    assert test_array.shape == (1, 4)
    assert np.isfinite(train_array).all()
    assert np.isfinite(test_array).all()

    assert metadata["numeric_columns"] == ["age", "spend"]
    assert metadata["categorical_columns"] == ["plan"]
    assert metadata["excluded_all_missing_columns"] == ["empty"]


def test_rejects_unsupported_datetime_columns():
    X_train = pd.DataFrame({
        "signup_date": pd.to_datetime(
            ["2025-01-01", "2025-01-02"]
        ),
    })

    with pytest.raises(ValueError, match="Unsupported feature types"):
        build_preprocessor(X_train)


def test_rejects_all_missing_features():
    X_train = pd.DataFrame({
        "age": [None, None],
    })

    with pytest.raises(ValueError, match="No usable"):
        build_preprocessor(X_train)


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataset"):
        diagnose_dataframe(pd.DataFrame())
