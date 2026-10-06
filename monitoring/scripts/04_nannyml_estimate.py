"""STEP 4 — NannyML: ประเมินประสิทธิภาพ "ตอนที่ยังไม่มีเฉลย" และตรวจ drift หลายตัวแปรพร้อมกัน

ปัญหาจริงของระบบ production คือเฉลย (label) มาช้า เช่น ต้องรออีก 30 วันจึงรู้ว่าลูกค้าเลิกใช้จริงไหม
ระหว่างนั้นเราจึงยังตอบไม่ได้ว่าโมเดลยังดีอยู่หรือเปล่า

CBPE (Confidence-Based Performance Estimation) แก้ปัญหานี้โดยใช้ค่าความมั่นใจของโมเดล
ประเมิน ROC AUC ล่วงหน้า  ข้อควรรู้ที่สำคัญที่สุดของ Lab นี้: CBPE ตั้งสมมติฐานว่า
"ความสัมพันธ์ระหว่าง input กับ label ไม่เปลี่ยน" ดังนั้นมันจับ Data Drift ได้ แต่ *จับ Concept Drift ไม่ได้*
"""
import json
import os

os.environ["NML_DISABLE_USAGE_LOGGING"] = "1"   # ปิดการส่งสถิติการใช้งานออกนอกเครื่อง

import nannyml as nml
import pandas as pd
from common import (
    ARTIFACTS,
    CATEGORICAL,
    FEATURES,
    REPORTS,
    TARGET,
    THRESHOLDS,
    ensure_dirs,
)

ensure_dirs()
ref = pd.read_csv(ARTIFACTS / "reference.csv", parse_dates=["timestamp"])
ana = pd.read_csv(ARTIFACTS / "production.csv", parse_dates=["timestamp"])

CHUNK = {"chunk_period": "W", "timestamp_column_name": "timestamp"}

# ---------- 4.1 ประเมินประสิทธิภาพโดยยังไม่ใช้เฉลย ----------
cbpe = nml.CBPE(
    y_pred_proba="y_proba",
    y_pred="y_pred",
    y_true=TARGET,
    problem_type="classification_binary",
    metrics=["roc_auc", "accuracy"],
    **CHUNK,
).fit(ref)                      # fit ใช้ reference ที่มีเฉลยครบ เพื่อ calibrate ค่าความมั่นใจ
est = cbpe.estimate(ana)        # estimate ใช้เฉพาะ y_proba/y_pred ของข้อมูล production
est.plot().write_html(str(REPORTS / "nannyml_estimated_performance.html"))

# ---------- 4.2 ประสิทธิภาพจริง (ใช้เปรียบเทียบเมื่อเฉลยกลับมาแล้ว) ----------
realized = nml.PerformanceCalculator(
    y_pred_proba="y_proba",
    y_pred="y_pred",
    y_true=TARGET,
    problem_type="classification_binary",
    metrics=["roc_auc", "accuracy"],
    **CHUNK,
).fit(ref).calculate(ana)
realized.plot().write_html(str(REPORTS / "nannyml_realized_performance.html"))

# ---------- 4.3 Multivariate drift — สรุป drift ของทุกฟีเจอร์เป็นตัวเลขเดียว ----------
# วิธี: ใช้ PCA บีบข้อมูลแล้วสร้างกลับ ถ้าโครงสร้างความสัมพันธ์ระหว่างฟีเจอร์เปลี่ยน
# ค่า reconstruction error จะสูงขึ้น แม้แต่ละฟีเจอร์เดี่ยว ๆ จะยังดูปกติ
multivar = nml.DataReconstructionDriftCalculator(column_names=FEATURES, **CHUNK).fit(ref).calculate(ana)
multivar.plot().write_html(str(REPORTS / "nannyml_multivariate_drift.html"))

# ---------- 4.4 Univariate drift — ดูรายฟีเจอร์ว่าใครเป็นต้นเหตุ ----------
univar = nml.UnivariateDriftCalculator(
    column_names=FEATURES,
    treat_as_categorical=CATEGORICAL,
    continuous_methods=["kolmogorov_smirnov"],
    categorical_methods=["chi2"],
    **CHUNK,
).fit(ref).calculate(ana)
univar.plot().write_html(str(REPORTS / "nannyml_univariate_drift.html"))

# ---------- 4.5 สรุปเป็นตารางเดียวเพื่อส่งต่อให้สคริปต์ตัดสินใจ ----------
est_df = est.filter(period="analysis").to_df()
real_df = realized.filter(period="analysis").to_df()
mv_df = multivar.filter(period="analysis").to_df()
weeks = [f"W{i + 1}" for i in range(len(est_df))]

rows = []
for i, week in enumerate(weeks):
    e_auc = float(est_df[("roc_auc", "value")].iloc[i])
    r_auc = float(real_df[("roc_auc", "value")].iloc[i])
    rows.append({
        "week": week,
        "estimated_roc_auc": round(e_auc, 4),
        "realized_roc_auc": round(r_auc, 4),
        "gap": round(r_auc - e_auc, 4),
        "estimated_alert": bool(est_df[("roc_auc", "alert")].iloc[i]),
        "multivariate_drift": round(float(mv_df[("reconstruction_error", "value")].iloc[i]), 4),
        "multivariate_alert": bool(mv_df[("reconstruction_error", "alert")].iloc[i]),
        "below_threshold": e_auc < THRESHOLDS["min_roc_auc"],
    })

(REPORTS / "nannyml_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print(pd.DataFrame(rows).to_string(index=False))
print("\nอ่านตารางนี้แบบนี้: gap ติดลบมาก = ค่าจริงแย่กว่าที่ประเมินไว้ = สัญญาณของ Concept Drift")
print("saved: reports/nannyml_*.html · reports/nannyml_summary.json")
