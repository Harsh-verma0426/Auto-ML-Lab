
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X_train: pd.DataFrame):
    """Build preprocessing steps using the training features.

    Returns:
        preprocessor: An unfitted scikit-learn ColumnTransformer.
        metadata: Information about the selected and excluded columns.
    """

    if X_train.empty:
        raise ValueError("Training features cannot be empty.")

    if not X_train.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")

    # Exclude columns that contain no observed training values.
    all_missing = X_train.columns[
        X_train.isna().all()
    ].tolist()

    usable = X_train.drop(columns=all_missing)

    # Identify supported feature types.
    numeric_columns = usable.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = usable.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    supported = set(numeric_columns + categorical_columns)

    unsupported_columns = [
        column for column in usable.columns
        if column not in supported
    ]

    if unsupported_columns:
        raise ValueError(
            "Unsupported feature types. Convert or handle these "
            f"columns explicitly: {unsupported_columns}"
        )

    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "No usable numerical or categorical features remain."
        )

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "excluded_all_missing_columns": all_missing,
        "unsupported_columns": unsupported_columns,
    }

    return preprocessor, metadata
