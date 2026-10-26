
import pandas as pd


def diagnose_dataframe(df: pd.DataFrame) -> dict:
    """Inspect common data-quality issues without modifying the data."""

    if df.empty:
        raise ValueError("Cannot diagnose an empty dataset.")

    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")

    columns = {}

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        observed = series.dropna()
        unique_count = int(observed.nunique())

        columns[str(column)] = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percent": round(
                missing_count / len(df) * 100, 2
            ),
            "unique_count": unique_count,
            "all_missing": missing_count == len(df),
            "constant": (
                unique_count == 1 and missing_count < len(df)
            ),
        }

    return {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }
