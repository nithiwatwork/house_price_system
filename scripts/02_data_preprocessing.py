import os

import pandas as pd
from sklearn.model_selection import train_test_split

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


def preprocess_data(test_size: float = 0.20, random_state: int = 42) -> str:
    """
    Loads raw data, splits it into training and testing sets,
    and logs the resulting datasets as artifacts in MLflow and processed_data/.
    """
    if not os.path.exists(DATA_PATH):
        from scripts.clean_dataset import download_dataset
        download_dataset(DATA_PATH)

    print(f"Starting data preprocessing on {DATA_PATH}...")

    # 1. Load data as a DataFrame
    df = pd.read_csv(DATA_PATH)

    # 2. Split the data into training and testing sets
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Create processed_data directory (รูปแบบ Lab 09)
    processed_data_dir = os.path.join(PROJECT_ROOT, "processed_data")
    os.makedirs(processed_data_dir, exist_ok=True)

    # Also save to data/processed for consistency
    data_processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(data_processed_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_data_dir, "test.csv"), index=False)
    train_df.to_csv(os.path.join(data_processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_processed_dir, "test.csv"), index=False)

    print(f"Saved processed data to '{processed_data_dir}' directory:")
    print(f"  - train.csv: {len(train_df):,} rows")
    print(f"  - test.csv: {len(test_df):,} rows")

    run_id = "local_preprocessing_run"

    # 4. Log to MLflow if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("House Price - Data Preprocessing")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.set_tag("ml.step", "data_preprocessing")
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_metric("training_set_rows", len(X_train))
            mlflow.log_metric("test_set_rows", len(X_test))
            mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
            print("Logged processed data as artifacts in MLflow.")

    print("-" * 50)
    print(f"Data preprocessing run finished. Preprocessing Run ID: {run_id}")
    print("-" * 50)
    return run_id


if __name__ == "__main__":
    preprocess_data()
