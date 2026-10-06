"""FastAPI service for house-price prediction."""

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter

from src.model import load_model


MODEL_PATH = Path("artifacts/serving_model/house_price_model.joblib")

app = FastAPI(title="House Price Prediction API")

# --- เพิ่ม Prometheus Metrics ---
# 1. สร้าง ASGI app สำหรับส่งออก metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 2. สร้าง Counter Metric สำหรับนับจำนวน request
PREDICT_REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["endpoint"])


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    # นับจำนวน Request เมื่อมีคนเรียกใช้ /predict
    PREDICT_REQUEST_COUNT.labels(endpoint="/predict").inc()

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not available.")

    model = load_model(MODEL_PATH)
    data = pd.DataFrame([request.features])
    prediction = model.predict(data)[0]

    return {"predicted_price": float(prediction)}