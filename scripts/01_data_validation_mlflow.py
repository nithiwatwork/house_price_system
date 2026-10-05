"""Validate prepared house-price data and log the result to MLflow."""

from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT /"data"/"raw"/"kc_house_data.csv"
TARGET_COLUMN = "price"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("House Price - Data Validation")

    with mlflow.start_run() as run:
        mlflow.set_tag("ml.step", "data_validation")

        if not DATA_PATH.exists():
            mlflow.log_param("validation_status", "Failed")
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

        data = pd.read_csv(DATA_PATH)
        rows = len(data)
        columns = len(data.columns)
        missing_values = int(data.isnull().sum().sum())
        target_exists = TARGET_COLUMN in data.columns

        mlflow.log_metric("num_rows", rows)
        mlflow.log_metric("num_columns", columns)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("target_exists", target_exists)

        validation_status = "Success"
        if not target_exists or rows == 0:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)

        print(f"Run ID: {run.info.run_id}")
        print(f"Rows: {rows}")
        print(f"Columns: {columns}")
        print(f"Missing values: {missing_values}")
        print(f"Validation status: {validation_status}")

        if validation_status == "Failed":
            raise SystemExit("Data validation failed.")


if __name__ == "__main__":
    main()
