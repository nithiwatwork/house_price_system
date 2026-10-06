"""STEP 3 — Evidently: ตรวจ Data Drift และคุณภาพการทำนายรายสัปดาห์

Evidently เปรียบเทียบ reference กับ current แล้วเลือกวิธีทดสอบทางสถิติให้เองตามชนิดและขนาดข้อมูล
(ตัวเลข -> Kolmogorov-Smirnov หรือ Wasserstein · หมวดหมู่ -> Chi-squared หรือ Jensen-Shannon)
ผลลัพธ์มี 2 รูปแบบ: HTML ให้คนอ่าน และ dict ให้โปรแกรมอ่านต่อเพื่อใช้เป็น "ด่านตรวจ"

แยกเป็น 3 ชั้น
  input drift      = การกระจายตัวของฟีเจอร์นำเข้าเปลี่ยน
  prediction drift = การกระจายตัวของคำทำนายเปลี่ยน (เห็นได้ทันทีแม้ยังไม่มีเฉลย)
  performance      = คุณภาพการทำนายจริง (ต้องรอเฉลยกลับมา)
"""
import json
import re

import pandas as pd
from common import (
    ARTIFACTS,
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    REPORTS,
    TARGET,
    THRESHOLDS,
    ensure_dirs,
)
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

# วิธีที่คืนค่า p-value จะ "drift" เมื่อค่าต่ำกว่า threshold ส่วนวิธีที่คืนระยะห่างจะกลับกัน
P_VALUE_METHODS = ("p_value", "chi-square", "z-test", "g-test", "fisher", "t-test", "anderson", "cramer")


def is_drifted(method: str, value: float, threshold: float) -> bool:
    if any(k in method.lower() for k in P_VALUE_METHODS):
        return value < threshold
    return value >= threshold


ensure_dirs()
ref = pd.read_csv(ARTIFACTS / "reference.csv")
prod = pd.read_csv(ARTIFACTS / "production.csv")

# บอก Evidently ให้ชัดว่าคอลัมน์ไหนคืออะไร ดีกว่าปล่อยให้เดาเอง
definition = DataDefinition(
    numerical_columns=NUMERIC + ["y_proba"],
    categorical_columns=CATEGORICAL + ["y_pred", TARGET],
    classification=[BinaryClassification(target=TARGET, prediction_labels="y_pred", prediction_probas="y_proba")],
)
COLS = FEATURES + [TARGET, "y_pred", "y_proba"]


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[COLS], data_definition=definition)


ref_ds = to_dataset(ref)
summary = []

for week, cur in prod.groupby("week"):
    report = Report([DataDriftPreset(drift_share=THRESHOLDS["drift_share"]), ClassificationPreset()])
    snapshot = report.run(current_data=to_dataset(cur), reference_data=ref_ds)

    # (1) รายงานสำหรับคนอ่าน — เปิดในเบราว์เซอร์เพื่อดูว่าฟีเจอร์ไหนเปลี่ยนและเปลี่ยนอย่างไร
    snapshot.save_html(str(REPORTS / f"evidently_{week}.html"))

    # (2) ผลแบบ dict สำหรับให้โปรแกรมตัดสินใจต่อ
    result = snapshot.dict()
    scores, drifted = {}, []
    for m in result["metrics"]:
        found = re.match(r"ValueDrift\(column=(.+?),method=(.+?),threshold=([\d.]+)\)", m["metric_name"])
        if not found:
            continue
        col, method, thr = found.group(1), found.group(2), float(found.group(3))
        scores[col] = (method, round(m["value"], 4))
        if is_drifted(method, m["value"], thr):
            drifted.append(col)

    def metric_of(name, res=result):
        return next(m["value"] for m in res["metrics"] if m["metric_name"].startswith(name))

    feature_drifted = [c for c in drifted if c in FEATURES]
    share = len(feature_drifted) / len(FEATURES)

    summary.append({
        "week": week,
        "n_rows": len(cur),
        "drift_share": round(share, 3),
        "drifted_features": feature_drifted,
        "prediction_drift": "y_proba" in drifted or "y_pred" in drifted,
        "roc_auc": round(metric_of("RocAuc("), 4),
        "accuracy": round(metric_of("Accuracy("), 4),
        "data_drift_alert": share > THRESHOLDS["drift_share"],
        "performance_alert": metric_of("RocAuc(") < THRESHOLDS["min_roc_auc"],
    })

(REPORTS / "evidently_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print(pd.DataFrame(summary)[
    ["week", "drift_share", "prediction_drift", "roc_auc", "accuracy", "data_drift_alert", "performance_alert"]
].to_string(index=False))
for row in summary:
    print(f"{row['week']} ฟีเจอร์ที่ drift: {', '.join(row['drifted_features']) or '(ไม่มี)'}")
print("\nsaved: reports/evidently_W1..W4.html · reports/evidently_summary.json")
