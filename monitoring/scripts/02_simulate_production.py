"""STEP 2 — จำลองข้อมูลที่ไหลเข้าระบบจริง 4 สัปดาห์

W1, W2  ปกติ                       -> ระบบควรเงียบ
W3      Data Drift (Covariate Shift) -> การกระจายตัวของ input เปลี่ยน
W4      Concept Drift                -> input เหมือนเดิม แต่ความสัมพันธ์ input -> label เปลี่ยน

จุดที่ต้องสังเกต: W4 จะ "หลุด" เครื่องมือที่ดูเฉพาะ input เพราะ input ไม่ได้เปลี่ยน
"""
import joblib
import numpy as np
import pandas as pd
from common import ARTIFACTS, FEATURES, TARGET, ensure_dirs

ensure_dirs()
rng = np.random.default_rng(42)
model = joblib.load(ARTIFACTS / "model.joblib")
pool = pd.read_csv(ARTIFACTS / "pool.csv")

WEEK_SIZE = len(pool) // 4
weeks = [pool.iloc[i * WEEK_SIZE:(i + 1) * WEEK_SIZE].copy() for i in range(4)]

# --- W3: Data Drift — ขึ้นราคา ลูกค้าใหม่เข้ามามาก และดันแพ็กเกจ Fiber optic ---
w3 = weeks[2]
w3["MonthlyCharges"] = (w3["MonthlyCharges"] * 1.25).round(2)
w3["tenure"] = (w3["tenure"] * 0.45).round().astype(int)
w3["TotalCharges"] = (w3["MonthlyCharges"] * np.maximum(w3["tenure"], 1)).round(2)
flip = rng.random(len(w3)) < 0.55
w3.loc[flip, "InternetService"] = "Fiber optic"

# --- W4: Concept Drift — input เหมือน W1/W2 แต่กฎการเลิกใช้บริการเปลี่ยน ---
# เดิม: ลูกค้าใหม่ + จ่ายแพง = เลิกใช้   ใหม่: คู่แข่งดึงลูกค้าเก่าสัญญายาวไป
w4 = weeks[3]
new_logit = (
    -0.4
    + 0.035 * w4["tenure"]
    - 0.015 * w4["MonthlyCharges"]
    + 1.0 * (w4["Contract"] != "Month-to-month")
)
new_label = (rng.random(len(w4)) < 1 / (1 + np.exp(-new_logit))).astype(int)
# กฎใหม่ยังไม่ได้ครอบคลุมลูกค้าทุกคน สมมติว่าเปลี่ยนไปแล้ว 70% ของฐานลูกค้า
switched = rng.random(len(w4)) < 0.70
w4[TARGET] = np.where(switched, new_label, w4[TARGET])

rows = []
for i, wk in enumerate(weeks, start=1):
    wk = wk.copy()
    wk["week"] = f"W{i}"
    wk["timestamp"] = pd.Timestamp("2026-09-07") + pd.to_timedelta(
        (i - 1) * 7 + rng.integers(0, 7, len(wk)), unit="D"
    )
    wk["y_proba"] = model.predict_proba(wk[FEATURES])[:, 1]
    wk["y_pred"] = (wk["y_proba"] >= 0.5).astype(int)
    rows.append(wk)

prod = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
prod.to_csv(ARTIFACTS / "production.csv", index=False)

print(prod.groupby("week").agg(
    n=("y_pred", "size"),
    mean_MonthlyCharges=("MonthlyCharges", "mean"),
    mean_tenure=("tenure", "mean"),
    pred_churn_rate=("y_pred", "mean"),
    actual_churn_rate=(TARGET, "mean"),
).round(3).to_string())
print("\nsaved: artifacts/production.csv")
