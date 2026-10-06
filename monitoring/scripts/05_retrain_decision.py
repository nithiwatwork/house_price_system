"""STEP 5 — ปิดวงจร: อ่านผลเฝ้าระวังทั้งหมด แล้วตัดสินใจว่าจะทำอะไรต่อ

นี่คือจุดที่ทำให้ "การเฝ้าระวัง" ต่างจาก "การทำกราฟสวย ๆ ไว้ดู"
ระบบต้องสรุปเป็นการกระทำได้ว่า  ปล่อยผ่าน / เฝ้าดูต่อ / เทรนใหม่ / ย้อนเวอร์ชัน

exit code ของสคริปต์นี้คือสิ่งที่ CI/CD หรือ orchestrator เอาไปใช้ต่อ
  0 = ปกติ        1 = ต้องเทรนใหม่
"""
import json
import sys

import pandas as pd
from common import REPORTS, THRESHOLDS

evidently = {r["week"]: r for r in json.loads((REPORTS / "evidently_summary.json").read_text())}
nannyml = {r["week"]: r for r in json.loads((REPORTS / "nannyml_summary.json").read_text())}

GAP_LIMIT = 0.10          # ค่าจริงต่ำกว่าค่าที่ประเมินเกินเท่านี้ = สงสัย Concept Drift
decisions = []

for week in sorted(evidently):
    ev, nm = evidently[week], nannyml[week]
    data_drift = ev["data_drift_alert"]
    realized = ev["roc_auc"]
    estimated = nm["estimated_roc_auc"]
    gap = round(estimated - realized, 4)
    perf_bad = realized < THRESHOLDS["min_roc_auc"]
    concept = gap > GAP_LIMIT

    if perf_bad and concept:
        action, reason = "RETRAIN", "Concept Drift — ความสัมพันธ์ input→label เปลี่ยน ต้องใช้เฉลยใหม่เทรน"
    elif perf_bad:
        action, reason = "RETRAIN", "ประสิทธิภาพต่ำกว่าเกณฑ์"
    elif data_drift:
        action, reason = "WATCH", "Data Drift แต่ประสิทธิภาพยังผ่านเกณฑ์ — เฝ้าดูและเตรียมข้อมูลใหม่ไว้"
    else:
        action, reason = "OK", "ทุกตัวชี้วัดอยู่ในเกณฑ์"

    decisions.append({
        "week": week, "drift_share": ev["drift_share"], "realized_roc_auc": realized,
        "estimated_roc_auc": estimated, "gap": gap, "action": action, "reason": reason,
    })

(REPORTS / "retrain_decision.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False))

df = pd.DataFrame(decisions)
print(df[["week", "drift_share", "realized_roc_auc", "estimated_roc_auc", "gap", "action"]].to_string(index=False))
print()
for d in decisions:
    print(f"{d['week']}: {d['action']} — {d['reason']}")

need_retrain = [d["week"] for d in decisions if d["action"] == "RETRAIN"]
if need_retrain:
    print(f"\n>>> ต้องเทรนใหม่ในสัปดาห์: {', '.join(need_retrain)}")
    print(">>> exit code 1 — ให้ pipeline รับไปสั่งงานเทรนใหม่ต่อ")
    sys.exit(1)
print("\n>>> ไม่ต้องเทรนใหม่ exit code 0")
