
import pandas as pd

from automl.data_quality import diagnose_dataframe
from automl.preprocessing import build_preprocessor


df = pd.read_csv("data/raw/sample_customers.csv")

# Inspect the original data.
report = diagnose_dataframe(df)

print("Dataset shape:", report["shape"])
print("Duplicate rows:", report["duplicate_rows"])

for column, info in report["columns"].items():
    print(
        column,
        "| missing:", info["missing_count"],
        "| constant:", info["constant"],
        "| all missing:", info["all_missing"],
    )

# Separate the target from the input features.
X = df.drop(columns=["churn"])

# Build a preprocessing plan without fitting it.
preprocessor, metadata = build_preprocessor(X)

print("\nPreprocessing plan:")
print(metadata)

print("\nPipeline:")
print(preprocessor)
