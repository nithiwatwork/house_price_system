import os
import sys

import joblib
import numpy as np
import pandas as pd

try:
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(
        f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db').replace(os.sep, '/')}"
    )
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME = "house-price-regressor-prod"
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "serving_model", "model.joblib")
LOCAL_PREP_PATH = os.path.join(PROJECT_ROOT, "artifacts", "serving_model", "preprocessor.joblib")


def load_and_predict():
    """Loads active registered model and runs inference on house sample."""
    sample = {
        "bedrooms": 3,
        "bathrooms": 2.25,
        "sqft_living": 2570,
        "sqft_lot": 7242,
        "floors": 2.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 2170,
        "sqft_basement": 400,
        "yr_built": 1951,
        "yr_renovated": 1991,
        "zipcode": 98125,
        "lat": 47.721,
        "long": -122.319,
        "sqft_living15": 1690,
        "sqft_lot15": 7639,
        "date": "20141209T000000"
    }

    df = pd.DataFrame([sample])

    # Try loading from MLflow Registry first, otherwise fallback to local artifact
    try:
        model_uri = f"models:/{MODEL_NAME}@staging"
        print(f"Loading registered model from MLflow: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        preprocessor = joblib.load(LOCAL_PREP_PATH)
        X_proc = preprocessor.transform(df)
        preds_log = model.predict(X_proc)
        prediction = np.expm1(preds_log)[0]
    except Exception:
        print(f"Loading model from local artifact: {LOCAL_MODEL_PATH}")
        model = joblib.load(LOCAL_MODEL_PATH)
        preprocessor = joblib.load(LOCAL_PREP_PATH)
        X_proc = preprocessor.transform(df)
        preds_log = model.predict(X_proc)
        prediction = np.expm1(preds_log)[0]

    print("=" * 60)
    print("           PREDICTION RESULT (04_load_and_predict)            ")
    print("=" * 60)
    print(f"Property Specs: {sample['bedrooms']} bed, {sample['bathrooms']} bath, {sample['sqft_living']} sqft, Grade {sample['grade']}")
    print(f"Valuation:      ${prediction:,.2f} USD")
    print("=" * 60)
    return prediction


if __name__ == "__main__":
    load_and_predict()
