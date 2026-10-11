import argparse
import json
from pathlib import Path

import pandas as pd

from automl.data_inspector import inspect_dataframe

def main():
    parser = argparse.ArgumentParser(
        description="Inspect a tabular dataset."
    )
    parser.add_argument(
        "dataset",
        help="Path to a CSV dataset"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.is_file():
        parser.error(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    report = inspect_dataframe(df)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()