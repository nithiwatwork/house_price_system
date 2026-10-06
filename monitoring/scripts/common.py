"""ค่าคงที่และฟังก์ชันที่ทุกสคริปต์ใน Lab 10 ใช้ร่วมกัน

จุดสำคัญเชิง MLOps: โค้ดแปลงข้อมูลอยู่ที่เดียวและถูกเรียกทั้งตอนเทรนและตอนให้บริการ
จึงกัน Training-Serving Skew ได้ตั้งแต่ต้น
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL

# เกณฑ์แจ้งเตือน ประกาศไว้ที่เดียว สคริปต์ตรวจและ Prometheus ใช้ค่าชุดเดียวกัน
THRESHOLDS = {
    "drift_share": 0.20,      # สัดส่วนฟีเจอร์ที่ drift เกินเท่านี้ถือว่าผิดปกติ
    "min_roc_auc": 0.78,      # AUC ต่ำกว่านี้ถือว่าโมเดลเสื่อม
    "p95_latency_ms": 300.0,  # SLO ของเวลาตอบกลับ
    "error_rate": 0.01,       # SLO ของอัตราข้อผิดพลาด
}


def load_clean_telco(csv_path=DATA_CSV) -> pd.DataFrame:
    """โหลด Telco Churn แล้วทำความสะอาด"""
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop(columns=["customerID"])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def ensure_dirs() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
