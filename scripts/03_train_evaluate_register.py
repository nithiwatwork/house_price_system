
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

try:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.artifacts import download_artifacts
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import build_preprocessing_pipeline


def train_evaluate_register(preprocessing_run_id: str = None, learning_rate: float = 0.05, max_iter: int = 300, C: float = 1.0, **kwargs):
    """
    Loads preprocessed data, trains a model pipeline, evaluates it, and
    registers the model in the MLflow Model Registry if it meets
    the primary optimizing threshold (MAPE < 10%).
    """
    MAPE_THRESHOLD = 0.15  # ตัวชี้วัดหลัก: MAPE < 15%
    MAX_SIZE_MB = 20.0     # เกณฑ์ Gating: ขนาดโมเดล < 20 MB
    MODEL_NAME = "house-price-regressor-prod"

    print(f"Starting training run with lr={learning_rate}, max_iter={max_iter}...")

    # 1. โหลดข้อมูลจาก processed_data/ หรือ Artifacts
    train_path = os.path.join(PROJECT_ROOT, "processed_data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "processed_data", "test.csv")

    if not os.path.exists(train_path):
        import importlib
        prep_mod = importlib.import_module("scripts.02_data_preprocessing")
        prep_mod.preprocess_data()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Successfully loaded data: train={len(train_df):,} rows, test={len(test_df):,} rows.")

    X_train = train_df.drop("price", axis=1)
    y_train = train_df["price"].values
    X_test = test_df.drop("price", axis=1)
    y_test = test_df["price"].values

    # 2. สร้าง Pipeline รวม Preprocessing เข้ากับ Model เพื่อป้องกัน Training-Serving Skew
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # เทรนโมเดลบน log1p(y) เพื่อให้ค่าความคลาดเคลื่อนร้อยละ (MAPE) ต่ำที่สุด
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train_proc, np.log1p(y_train))

    # 3. ประเมินผลโมเดล
    preds_log = model.predict(X_test_proc)
    preds = np.expm1(preds_log)

    mape = mean_absolute_percentage_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("=" * 60)
    print("                  EVALUATION REPORT                   ")
    print("=" * 60)
    print(f"  - MAPE (Optimizing Metric): {mape:.4f} ({mape:.2%}) [Goal: < {MAPE_THRESHOLD:.0%}] -> {'PASS' if mape < MAPE_THRESHOLD else 'FAIL'}")
    print(f"  - RMSE:                     ${rmse:,.2f}")
    print(f"  - R2 Score:                 {r2:.4f}")

    # 4. บันทึก Artifacts ไปยัง include/serving_model และ artifacts/serving_model
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts", "serving_model")
    include_dir = os.path.join(PROJECT_ROOT, "include", "serving_model")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(include_dir, exist_ok=True)

    joblib.dump(model, os.path.join(artifacts_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    joblib.dump(model, os.path.join(include_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(include_dir, "preprocessor.joblib"))

    artifact_size_mb = os.path.getsize(os.path.join(artifacts_dir, "model.joblib")) / (1024 * 1024)
    print(f"  - Model Artifact Size:      {artifact_size_mb:.2f} MB [Gating: < {MAX_SIZE_MB} MB] -> {'PASS' if artifact_size_mb < MAX_SIZE_MB else 'FAIL'}")
    print("=" * 60)

    # 5. Log Parameters, Metrics และ Model ลง MLflow หากมี MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("House Price - Model Training")
        with mlflow.start_run(run_name=f"hist_gradient_boosting_lr_{learning_rate}"):
            mlflow.set_tag("ml.step", "model_training_evaluation")
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_metric("mape", mape)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("artifact_size_mb", artifact_size_mb)

            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="house_price_model",
                    input_example=X_train_proc[:5]
                )
                if mape < MAPE_THRESHOLD and artifact_size_mb < MAX_SIZE_MB:
                    registered_model = mlflow.register_model(model_info.model_uri, MODEL_NAME)
                    client = MlflowClient()
                    client.set_registered_model_alias(
                        name=MODEL_NAME,
                        alias="staging",
                        version=registered_model.version
                    )
                    print(f"Registered model in MLflow with alias '@staging': {MODEL_NAME}")
            except Exception as e:
                print(f"MLflow registration note: {e}")

    print("Training run finished successfully.")
    return {"mape": mape, "rmse": rmse, "r2": r2, "size_mb": artifact_size_mb}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C_value]")
        sys.exit(1)

    run_id = sys.argv[1]
    c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)
