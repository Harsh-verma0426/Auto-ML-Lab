import pandas as pd

def infer_column_type(series: pd.Series):

    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"

    if (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
        return "text_or_mixed"

    return "other"


def inspect_dataframe(df: pd.DataFrame) -> dict:

    if df.empty:
        raise ValueError("Cannot inspect a dataset with no rows.")

    column_profiles = {}

    for column in df.columns:
        series = df[column]

        missing_count = int(series.isna().sum())
        non_missing = series.dropna()
        unique_count = int(non_missing.nunique())

        column_profiles[str(column)] = {
            "dtype": str(series.dtype),
            "inferred_type": infer_column_type(series),
            "missing_count": missing_count,
            "missing_percent": round((missing_count / len(df)) * 100, 2),
            "unique_count" : unique_count,
            "all_missing": missing_count == len(df),
            "is_constant": (unique_count==1 and missing_count < len(df))
        }

    return {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": column_profiles
    }