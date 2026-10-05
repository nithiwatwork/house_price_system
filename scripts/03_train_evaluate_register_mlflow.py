"""Train, evaluate, and register the house-price model with MLflow."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "house-price-model"
MODEL_ALIAS = "staging"
N_ESTIMATORS = 100
RANDOM_STATE = 42


def evaluate(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions) ** 0.5
    r2 = r2_score(y_true, predictions)
    return mae, rmse, r2


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("House Price - Model Training")

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    x_train = train.drop(columns=["price"])
    y_train = train["price"]
    x_validation = validation.drop(columns=["price"])
    y_validation = validation["price"]
    x_test = test.drop(columns=["price"])
    y_test = test["price"]

    model = build_pipeline(x_train)

    with mlflow.start_run(run_name="random_forest_house_price") as run:
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.log_params(
            {
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "target_column": "price",
            }
        )

        model.fit(x_train, y_train)

        validation_predictions = model.predict(x_validation)
        validation_mae, validation_rmse, validation_r2 = evaluate(
            y_validation, validation_predictions
        )

        test_predictions = model.predict(x_test)
        test_mae, test_rmse, test_r2 = evaluate(y_test, test_predictions)

        mlflow.log_metrics(
            {
                "validation_mae": validation_mae,
                "validation_rmse": validation_rmse,
                "validation_r2": validation_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }
        )

        signature = infer_signature(
            x_train, model.predict(x_train.head(5))
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="house_price_pipeline",
            input_example=x_train.head(5),
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        client = MlflowClient()
        latest_version = client.get_registered_model(
            MODEL_NAME
        ).latest_versions[-1].version

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS,
            version=latest_version,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Validation MAE: {validation_mae:.4f}")
        print(f"Validation RMSE: {validation_rmse:.4f}")
        print(f"Validation R2: {validation_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R2: {test_r2:.4f}")
        print(f"Model URI: {model_info.model_uri}")
        print(
            f"Registered model: {MODEL_NAME} "
            f"version {latest_version} -> @{MODEL_ALIAS}"
        )


if __name__ == "__main__":
    main()
