"""Log the prepared train/validation/test splits as MLflow artifacts."""

from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("House Price - Data Preprocessing")

    required_files = ["train.csv", "validation.csv", "test.csv"]

    with mlflow.start_run() as run:
        mlflow.set_tag("ml.step", "data_preprocessing")

        for filename in required_files:
            file_path = PROCESSED_DIR / filename

            if not file_path.exists():
                raise FileNotFoundError(
                    f"{file_path} not found. Run scripts/00_data_ingestion.py first."
                )

            data = pd.read_csv(file_path)
            mlflow.log_metric(filename.replace(".csv", "_rows"), len(data))
            mlflow.log_artifact(str(file_path), artifact_path="processed_data")

        mlflow.log_param("processed_data_dir", str(PROCESSED_DIR))
        print(f"Preprocessing Run ID: {run.info.run_id}")
        print("Logged train.csv, validation.csv and test.csv to MLflow.")


if __name__ == "__main__":
    main()
