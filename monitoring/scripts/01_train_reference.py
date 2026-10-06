"""STEP 1 — เทรนโมเดลฐาน และสร้าง reference set

reference set คือ "ภาพจำ" ของระบบตอนที่ยังทำงานดี ทุกเครื่องมือเฝ้าระวังใน Lab นี้
เปรียบเทียบข้อมูลปัจจุบันกับชุดนี้เสมอ จึงต้องเก็บ features, y_true, y_pred และ y_proba ไว้ครบ
"""
import joblib
import numpy as np
import pandas as pd
from common import (
    ARTIFACTS,
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    TARGET,
    ensure_dirs,
    load_clean_telco,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ensure_dirs()
df = load_clean_telco()

# แบ่ง 3 ส่วน: train เทรนโมเดล · reference ใช้เป็นฐานเปรียบเทียบ · pool เก็บไว้จำลอง production
train_df, rest = train_test_split(df, test_size=0.5, random_state=42, stratify=df[TARGET])
ref_df, pool_df = train_test_split(rest, test_size=0.6, random_state=42, stratify=rest[TARGET])

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
    remainder="passthrough",
)
model = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=42))])
model.fit(train_df[FEATURES], train_df[TARGET])

ref_out = ref_df.copy()
# ให้ reference มีแกนเวลาด้วย (4 สัปดาห์ก่อนขึ้นระบบ) เพราะ NannyML ใช้ timestamp ในการแบ่ง chunk
rng = np.random.default_rng(0)
ref_out["timestamp"] = pd.Timestamp("2026-08-03") + pd.to_timedelta(rng.integers(0, 28, len(ref_out)), unit="D")
ref_out["y_proba"] = model.predict_proba(ref_df[FEATURES])[:, 1]
ref_out["y_pred"] = (ref_out["y_proba"] >= 0.5).astype(int)
ref_auc = roc_auc_score(ref_out[TARGET], ref_out["y_proba"])

joblib.dump(model, ARTIFACTS / "model.joblib")
ref_out.to_csv(ARTIFACTS / "reference.csv", index=False)
pool_df.to_csv(ARTIFACTS / "pool.csv", index=False)

print(f"train={len(train_df)} reference={len(ref_out)} pool={len(pool_df)}")
print(f"reference ROC AUC = {ref_auc:.4f}   (ค่านี้คือเกณฑ์ที่ใช้เทียบตอนโมเดลขึ้นระบบ)")
print(f"numeric={NUMERIC}")
print("saved: artifacts/model.joblib · artifacts/reference.csv · artifacts/pool.csv")
