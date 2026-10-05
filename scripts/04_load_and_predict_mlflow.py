"""Load the registered house-price model by MLflow 3 alias and predict."""

from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_NAME = "house-price-model"
MODEL_ALIAS = "staging"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    test = pd.read_csv(TEST_PATH)
    sample = test.drop(columns=["price"]).head(1)

    prediction = model.predict(sample)[0]
    actual = test["price"].iloc[0]

    print("-" * 40)
    print(f"Actual price: {actual}")
    print(f"Predicted price: {prediction}")
    print("-" * 40)


if __name__ == "__main__":
    main()
