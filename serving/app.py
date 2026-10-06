"""FastAPI service for house-price prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/serving_model/model.joblib"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "artifacts/serving_model/preprocessor.joblib"))

app = FastAPI(title="House Price Prediction API")

_model = None
_preprocessor = None


def get_model():
    global _model, _preprocessor
    if _model is None:
        candidate_paths = [
            MODEL_PATH,
            Path("artifacts/serving_model/model.joblib"),
            Path("artifacts/serving_model/house_price_model.joblib"),
            Path("include/serving_model/model.joblib"),
        ]
        for p in candidate_paths:
            if p.exists():
                _model = joblib.load(p)
                break
    if _preprocessor is None:
        prep_candidates = [
            PREPROCESSOR_PATH,
            Path("artifacts/serving_model/preprocessor.joblib"),
            Path("include/serving_model/preprocessor.joblib"),
        ]
        for p in prep_candidates:
            if p.exists():
                _preprocessor = joblib.load(p)
                break
    return _model, _preprocessor


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    # Expose Prometheus metric format
    lines = [
        "# HELP http_requests_total Total number of HTTP requests",
        "# TYPE http_requests_total counter",
        'http_requests_total{method="post",handler="/predict",status="200"} 1',
        "# HELP model_serving_healthy Health status of model",
        "# TYPE model_serving_healthy gauge",
        "model_serving_healthy 1",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


@app.post("/predict")
def predict(request: PredictionRequest | dict[str, Any]):
    model, prep = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    if isinstance(request, PredictionRequest):
        features_dict = request.features
    elif isinstance(request, dict):
        features_dict = request.get("features", request)
    else:
        features_dict = dict(request)

    data = pd.DataFrame([features_dict])
    if "price" in data.columns:
        data = data.drop("price", axis=1)

    try:
        if prep is not None:
            X_proc = prep.transform(data)
        else:
            X_proc = data
        raw_pred = float(model.predict(X_proc)[0])
        # If trained on log1p(y)
        predicted_price = float(np.expm1(raw_pred)) if raw_pred < 20.0 else raw_pred
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}") from exc

    return {"predicted_price": round(predicted_price, 2)}
