import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def _json_safe(value):

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if value is pd.NA or value is pd.NaT:
        return None

    return value


def _save_plot(fig, output_dir: Path, filename: str):

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_eda(
        df: pd.DataFrame,
        output_dir: str | Path,
        target: str | None = None,
) -> dict:

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if df.shape[1] == 0:
        raise ValueError("Dataset has no columns.")

    if not df.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    if target is not None and target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    report = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "missing_values": {
            "count_by_column": df.isna().sum().to_dict(),
            "percent_by_column": (df.isna().mean().mul(100).round(2).to_dict()),
            "total_cells_missing": int(df.isna().sum().sum()),
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": (numeric.describe().round(3).to_dict()),
        "categorical_summary": {},
        "correlation": {},
        "target": target,
        "plots": [],
    }

    for column in numeric.columns:
        values = numeric[column].dropna()

        if values.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(values, kde=False, ax=ax)
        ax.set_title(f"Distribution: {column}")
        ax.set_xlabel(str(column))

        filename = f"distribution_{len(report['plots']) + 1}.png"
        _save_plot(fig, output_dir, filename)
        report["plots"].append(filename)


    for column in categorical.columns:
        counts = df[column].value_counts(dropna=False)

        report["categorical_summary"][str(column)] = {
            str(k): int(v) for k, v in counts.items()
        }

        if counts.empty:
            continue

        plot_counts = counts.head(20).copy()
        labels = [
            "<MISSING>" if pd.isna(v) else str(v)
            for v in plot_counts.index
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, plot_counts.values)
        ax.set_title(f"Category counts: {column}")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=45)

        filename = f"categories_{len(report['plots']) + 1}.png"
        _save_plot(fig, output_dir, filename)
        report["plots"].append(filename)

    
    if numeric.shape[1] >= 2:
        correlation = numeric.corr()
        report["correlation"] = correlation.round(4).to_dict()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
        )
        ax.set_title("Numerical feature correlations")

        filename = "correlation_heatmap.png"
        _save_plot(fig, output_dir, filename)
        report["plots"].append(filename)

    missing_percent = df.isna().mean().mul(100)
    missing_percent = missing_percent[missing_percent > 0]

    if not missing_percent.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        missing_percent.sort_values().plot.barh(ax=ax)
        ax.set_xlabel("Missing values (%)")
        ax.set_title("Missingness by column")

        filename = "missing_values.png"
        _save_plot(fig, output_dir, filename)
        report["plots"].append(filename)

    if target is not None:
        target_series = df[target]

        report["target_summary"] = {
            "non_missing": int(target_series.notna().sum()),
            "missing": int(target_series.isna().sum()),
            "unique_values": int(target_series.nunique()),
        }

        if (
            not pd.api.types.is_numeric_dtype(target_series)
            or target_series.nunique(dropna=True) <= 20
        ):
            counts = target_series.value_counts(dropna=False)

            fig, ax = plt.subplots(figsize=(7, 4))
            labels = [
                "<MISSING>" if pd.isna(v) else str(v)
                for v in counts.index
            ]
            ax.bar(labels, counts.values)
            ax.set_title(f"Target distribution: {target}")
            ax.tick_params(axis="x", labelrotation=45)

            filename = "target_distribution.png"
            _save_plot(fig, output_dir, filename)
            report["plots"].append(filename)

    safe_report = _json_safe(report)

    with (output_dir / "eda_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(safe_report, file, indent=2, allow_nan=False)

    return safe_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate an EDA report for a CSV dataset."
    )
    parser.add_argument("dataset", help="Path to a CSV file")
    parser.add_argument(
        "--output",
        default="reports/eda",
        help="Directory for report and plots",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column name",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.is_file():
        parser.error(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    report = generate_eda(
        df,
        output_dir=args.output,
        target=args.target,
    )

    print(f"EDA complete: {args.output}")
    print(f"Shape: {report['shape']}")
    print(f"Plots generated: {len(report['plots'])}")
    print(f"Report: {Path(args.output) / 'eda_report.json'}")


if __name__ == "__main__":
    main()