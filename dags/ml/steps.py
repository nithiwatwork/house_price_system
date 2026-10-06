"""Step implementations for House Price Airflow Pipeline following TFX-style architecture."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_preprocessing_pipeline

EXPECTED_COLUMNS = [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]


def _find_data_file(data_file: str) -> Path:
    """Find data file path across common locations."""
    candidates = [
        Path(data_file),
        PROJECT_ROOT / data_file,
        PROJECT_ROOT / "data" / "raw" / data_file,
        PROJECT_ROOT / "data" / "corrupted" / data_file,
        PROJECT_ROOT / "data" / data_file,
        Path("/opt/airflow/data/raw") / data_file,
        Path("/opt/airflow/data/corrupted") / data_file,
        Path("/opt/airflow/data") / data_file,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    # Default to raw kc_house_data.csv if exists
    default_path = PROJECT_ROOT / "data" / "raw" / "kc_house_data.csv"
    if default_path.exists():
        return default_path
    raise FileNotFoundError(f"Could not locate data file: {data_file}")


def example_gen(run_id: str, data_file: str = "kc_house_data.csv") -> dict[str, Any]:
    """Ingest dataset and perform reproducible train/test partition."""
    source_path = _find_data_file(data_file)
    df = pd.read_csv(source_path)

    run_dir = PROJECT_ROOT / "include" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "raw_data.csv"
    train_path = run_dir / "train.csv"
    test_path = run_dir / "test.csv"

    df.to_csv(raw_path, index=False)

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Sync to processed_data directory
    processed_dir = PROJECT_ROOT / "processed_data"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(f"[ExampleGen] Ingested {len(df):,} rows from {source_path}")
    print(f"[ExampleGen] Train: {len(train_df):,} rows -> {train_path}")
    print(f"[ExampleGen] Test:  {len(test_df):,} rows -> {test_path}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "raw_path": str(raw_path),
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "data_file": data_file,
    }


def statistics_gen(run_id: str, examples: dict[str, Any]) -> str:
    """Generate statistical profile for the ingested training data."""
    train_df = pd.read_csv(examples["train_path"])
    run_dir = Path(examples["run_dir"])

    stats = {
        "row_count": len(train_df),
        "column_count": len(train_df.columns),
        "numeric_summary": train_df.describe().to_dict(),
        "missing_counts": train_df.isnull().sum().to_dict(),
    }

    stats_path = run_dir / "statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"[StatisticsGen] Summary saved to {stats_path}")
    return str(stats_path)


def schema_gen(run_id: str, statistics_path: str) -> str:
    """Infer and emit data schema contract and validation constraints."""
    run_dir = Path(statistics_path).parent

    schema_contract = {
        "expected_columns": EXPECTED_COLUMNS,
        "constraints": {
            "price": {"min": 1.0, "nullable": False},
            "bedrooms": {"min": 0, "max": 50, "nullable": False},
            "sqft_living": {"min": 100, "nullable": False},
        },
    }

    schema_path = run_dir / "schema.json"
    schema_path.write_text(json.dumps(schema_contract, indent=2), encoding="utf-8")
    print(f"[SchemaGen] Schema definition saved to {schema_path}")
    return str(schema_path)


def example_validator(examples: dict[str, Any], schema_path: str) -> dict[str, Any]:
    """Validate data against schema contracts and identify anomalies."""
    train_df = pd.read_csv(examples["train_path"])
    anomalies: list[str] = []

    # 1. Missing columns check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in train_df.columns]
    if missing_cols:
        anomalies.append(f"Missing required columns: {missing_cols}")

    # 2. Check for negative or zero price
    if "price" in train_df.columns:
        invalid_price = int((train_df["price"] <= 0).sum())
        if invalid_price > 0:
            anomalies.append(f"Invalid price (<=0) found in {invalid_price} rows")

    # 3. Check for impossible bedroom counts
    if "bedrooms" in train_df.columns:
        bad_bedrooms = int((train_df["bedrooms"] > 50).sum())
        if bad_bedrooms > 0:
            anomalies.append(f"Impossible bedroom count (>50) found in {bad_bedrooms} rows")

    # 4. Check for missing values in critical features
    null_counts = train_df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        anomalies.append(f"Missing values detected: {cols_with_nulls}")

    ok = len(anomalies) == 0
    print(f"[ExampleValidator] Validation status: {'PASS' if ok else 'FAIL'}")
    return {
        "ok": ok,
        "anomalies": anomalies,
    }


def transform(run_id: str, examples: dict[str, Any]) -> str:
    """Preprocess features using TabularPreprocessor learned from training data only."""
    train_df = pd.read_csv(examples["train_path"])
    test_df = pd.read_csv(examples["test_path"])

    run_dir = Path(examples["run_dir"])
    transform_dir = run_dir / "transform"
    transform_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df.drop("price", axis=1)
    y_train = train_df["price"].values
    X_test = test_df.drop("price", axis=1)
    y_test = test_df["price"].values

    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    joblib.dump(preprocessor, transform_dir / "preprocessor.joblib")
    joblib.dump(X_train_proc, transform_dir / "X_train.joblib")
    joblib.dump(y_train, transform_dir / "y_train.joblib")
    joblib.dump(X_test_proc, transform_dir / "X_test.joblib")
    joblib.dump(y_test, transform_dir / "y_test.joblib")

    print(f"[Transform] Feature transform completed -> {transform_dir}")
    return str(transform_dir)


def data_card_gen(run_id: str, examples: dict[str, Any]) -> str:
    """Generate data profile card in parallel with model training."""
    train_df = pd.read_csv(examples["train_path"])
    run_dir = Path(examples["run_dir"])

    card = {
        "dataset_name": "King County House Sales",
        "run_id": run_id,
        "total_records": examples["num_rows"],
        "training_records": len(train_df),
        "features": list(train_df.columns),
        "target": "price",
        "price_summary": {
            "min": float(train_df["price"].min()) if "price" in train_df.columns else None,
            "median": float(train_df["price"].median()) if "price" in train_df.columns else None,
            "max": float(train_df["price"].max()) if "price" in train_df.columns else None,
        },
    }

    card_path = run_dir / "data_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"[DataCard] Data profile card generated -> {card_path}")
    return str(card_path)


def trainer(run_id: str, examples: dict[str, Any], transform_path: str, max_iter: int = 200) -> str:
    """Train HistGradientBoostingRegressor on preprocessed training data."""
    transform_dir = Path(transform_path)
    X_train_proc = joblib.load(transform_dir / "X_train.joblib")
    y_train = joblib.load(transform_dir / "y_train.joblib")

    model = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
    )
    # Train on log1p(y) to minimize percentage error (MAPE)
    model.fit(X_train_proc, np.log1p(y_train))

    run_dir = Path(examples["run_dir"])
    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"[Trainer] Model successfully trained and saved -> {model_path}")
    return str(model_path)


def evaluator(run_id: str, examples: dict[str, Any], model_path: str) -> dict[str, Any]:
    """Evaluate model performance against baseline and quality gating rules."""
    run_dir = Path(model_path).parent
    transform_dir = run_dir / "transform"

    model = joblib.load(model_path)
    X_test_proc = joblib.load(transform_dir / "X_test.joblib")
    y_test = joblib.load(transform_dir / "y_test.joblib")

    preds_log = model.predict(X_test_proc)
    preds = np.expm1(preds_log)

    mape = float(mean_absolute_percentage_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    baseline_mape = 0.20  # 20% baseline
    improvement = baseline_mape - mape

    # 3 Blessing Conditions:
    # 1. MAPE < 15% (Primary optimizing metric)
    # 2. R2 >= 0.70
    # 3. Model size < 20 MB
    blessed = bool(mape < 0.15 and r2 >= 0.70 and size_mb < 20.0)

    print("=" * 60)
    print(f"[Evaluator] MAPE: {mape:.4f} (Goal < 0.15) -> {'PASS' if mape < 0.15 else 'FAIL'}")
    print(f"[Evaluator] RMSE: ${rmse:,.2f}")
    print(f"[Evaluator] R2:   {r2:.4f} (Goal >= 0.70) -> {'PASS' if r2 >= 0.70 else 'FAIL'}")
    print(f"[Evaluator] Size: {size_mb:.2f} MB (Goal < 20 MB)")
    print(f"[Evaluator] Blessing Gate: {'BLESSED' if blessed else 'REJECTED'}")
    print("=" * 60)

    return {
        "blessed": blessed,
        "mape": round(mape, 4),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "baseline_mape": baseline_mape,
        "improvement": round(improvement, 4),
        "size_mb": round(size_mb, 2),
        "transform_dir": str(transform_dir),
    }


def pusher(run_id: str, model_path: str, metrics: dict[str, Any], schema_path: str) -> str:
    """Deploy model and preprocessor to serving directories and MLflow Registry."""
    run_dir = Path(model_path).parent
    transform_dir = Path(metrics.get("transform_dir", run_dir / "transform"))

    artifacts_serving = PROJECT_ROOT / "artifacts" / "serving_model"
    include_serving = PROJECT_ROOT / "include" / "serving_model"
    artifacts_serving.mkdir(parents=True, exist_ok=True)
    include_serving.mkdir(parents=True, exist_ok=True)

    # 1. Deploy model & preprocessor to serving
    shutil.copy2(model_path, artifacts_serving / "model.joblib")
    shutil.copy2(model_path, include_serving / "model.joblib")
    if (transform_dir / "preprocessor.joblib").exists():
        shutil.copy2(transform_dir / "preprocessor.joblib", artifacts_serving / "preprocessor.joblib")
        shutil.copy2(transform_dir / "preprocessor.joblib", include_serving / "preprocessor.joblib")

    # 2. Log & Register with MLflow if available
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        mlflow_db = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{str(mlflow_db).replace(os.sep, '/')}")
        mlflow.set_experiment("House Price - Airflow Pipeline")
        with mlflow.start_run(run_name=f"airflow_{run_id}"):
            mlflow.log_metrics({
                "mape": metrics["mape"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            })
            model = joblib.load(model_path)
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="house_price_model",
            )
            reg = mlflow.register_model(model_info.model_uri, "house-price-regressor-prod")
            client = MlflowClient()
            client.set_registered_model_alias("house-price-regressor-prod", "staging", reg.version)
            print(f"[Pusher] Model registered in MLflow version {reg.version} with @staging alias")
    except Exception as e:  # noqa: BLE001
        print(f"[Pusher] MLflow registration note: {e}")

    print("[Pusher] Model successfully blessed and pushed to serving!")
    return f"Model pushed successfully (MAPE: {metrics.get('mape')})"
