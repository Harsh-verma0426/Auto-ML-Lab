# Auto-ML-Lab
An independent project exploring automated machine learning,
starting with tabular data.

## Current capabilities

- CSV loading
- Basic column-type inference
- Missing-value profiling
- Duplicate-row detection
- Constant-column detection
- Automated dataset summary
- Automated tests

## Development setup

Requires Python 3.14 or newer and [uv](https://docs.astral.sh/uv/).
From the repository root, install the project and its development dependencies:

    uv sync --extra dev

Run the CLI and tests in the project environment:

    uv run python -m automl data/raw/sample_customers.csv
    uv run python -m pytest -v

Alternatively, install with pip into the active Python environment:

    python -m pip install -e ".[dev]"
    python -m automl data/raw/sample_customers.csv
    python -m pytest -v
