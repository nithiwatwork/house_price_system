"""FastAPI service for house-price prediction."""

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import load_model

MODEL_PATH = Path("artifacts/serving_model/house_price_model.joblib")

app = FastAPI(title="House Price Prediction API")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not available.")

    model = load_model(MODEL_PATH)
    data = pd.DataFrame([request.features])
    prediction = model.predict(data)[0]

    return {"predicted_price": float(prediction)}
