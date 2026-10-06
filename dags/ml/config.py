"""ค่าคงที่และเส้นทางไฟล์ของ pipeline

แยกออกมาจากไฟล์ DAG เพราะ DAG ควรบอกแค่ "ลำดับงาน" ส่วน "งานทำอะไร" อยู่ในโมดูลนี้กับ steps.py
ข้อดีคือทดสอบตรรกะได้โดยไม่ต้องยก Airflow ขึ้นมาทั้งระบบ
"""

import os
from pathlib import Path

# ภายใน container ของ Airflow โฟลเดอร์โปรเจกต์ถูก mount ไว้ที่ /opt/airflow
ROOT = Path(os.getenv("ML_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_CSV = ROOT / "data" / "raw" / "kc_house_data.csv"
INCLUDE = ROOT / "include"          # ที่เก็บ artifact ทั้งหมดที่ pipeline สร้าง
RUNS = INCLUDE / "runs"             # artifact แยกตาม run — นี่คือหัวใจของการทำซ้ำได้
SERVING = INCLUDE / "serving_model" # โมเดลที่ผ่านด่านแล้วเท่านั้นที่มาอยู่ตรงนี้
ARTIFACTS = ROOT / "artifacts" / "serving_model"

TARGET = "price"
NUMERIC = [
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
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]
CATEGORICAL = ["zipcode"]
FEATURES = NUMERIC + CATEGORICAL

# ด่านตรวจก่อนอนุมัติโมเดลขึ้นใช้งาน (Evaluator blessing)
MAX_MAPE = 0.15              # เกณฑ์ขั้นต่ำ MAPE < 15% (Primary optimizing metric)
MIN_R2 = 0.70                # R2 >= 0.70
MAX_MODEL_SIZE_MB = 20.0     # ขนาดโมเดล < 20 MB
BASELINE_MAPE = 0.20         # โมเดลเดิม/baseline 20%


def run_dir(run_id: str) -> Path:
    """โฟลเดอร์ artifact ของการรันหนึ่งครั้ง

    ใช้ run_id ของ Airflow เป็นชื่อ ทำให้ย้อนกลับไปดูได้เสมอว่าโมเดลตัวไหนมาจากการรันไหน
    และรันซ้ำด้วย run_id เดิมจะเขียนทับที่เดิม ไม่ปนกับรันอื่น (idempotent)
    """
    safe = run_id.replace(":", "-").replace("+", "_")
    d = RUNS / safe
    d.mkdir(parents=True, exist_ok=True)
    return d
