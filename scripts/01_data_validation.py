import os
import sys

import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(
        f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db').replace(os.sep, '/')}"
    )

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "kc_house_data.csv")


def validate_data(data_path: str = DATA_PATH):
    """
    Loads the King County dataset, performs basic validation checks,
    and logs the results to MLflow.
    """

    # 1. Load data as a Pandas DataFrame
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully from {data_path}")

    # 2. Perform validation checks
    num_rows, num_cols = df.shape
    missing_values = df.isnull().sum().sum()
    min_price = float(df["price"].min()) if "price" in df.columns else 0.0
    max_price = float(df["price"].max()) if "price" in df.columns else 0.0

    print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
    print(f"Missing values: {missing_values}")
    print(f"Price range: ${min_price:,.2f} - ${max_price:,.2f}")

    # Check if the data passes our defined criteria
    validation_status = "Success"
    if missing_values > 0 or num_rows < 20000 or min_price <= 0:
        validation_status = "Failed"

    print(f"Validation status: {validation_status}")

    # 3. Log validation results to MLflow if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("House Price - Data Validation")
        with mlflow.start_run():
            mlflow.set_tag("ml.step", "data_validation")
            mlflow.log_metric("num_rows", num_rows)
            mlflow.log_metric("num_cols", num_cols)
            mlflow.log_metric("missing_values", missing_values)
            mlflow.log_metric("min_price", min_price)
            mlflow.log_metric("max_price", max_price)
            mlflow.log_param("validation_status", validation_status)

    # 4. ทำให้ CI จับได้จริง — ต้องคืน exit code ที่ไม่ใช่ 0 เมื่อข้อมูลไม่ผ่าน
    if validation_status == "Failed":
        raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")

    print("Data validation run finished successfully.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    validate_data(target)
