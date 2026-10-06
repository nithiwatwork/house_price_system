"""สะพานเชื่อมงานแบบ batch (Evidently/NannyML) เข้ากับ Prometheus

Prometheus ใช้วิธี "ดึง" (pull) คือวิ่งมาขอค่าที่ endpoint /metrics เป็นระยะ
แต่การคำนวณ drift เป็นงาน batch ที่รันวันละครั้ง เราจึงต้องมีตัวกลางที่
อ่านผลล่าสุดจากไฟล์ JSON แล้วประกาศเป็น metric ค้างไว้ให้ Prometheus มาดึง

รันแยกพอร์ตจากบริการทำนาย เพราะเป็นคนละวงจรชีวิตกัน
"""
import json
import time
from pathlib import Path

from prometheus_client import Gauge, start_http_server

REPORTS = Path("reports")
PORT = 9101
REFRESH_SEC = 15

DRIFT_SHARE = Gauge("ml_data_drift_share", "สัดส่วนฟีเจอร์ที่ drift เทียบกับ reference", ["week"])
DRIFTED_N = Gauge("ml_drifted_features", "จำนวนฟีเจอร์ที่ drift", ["week"])
REALIZED_MAPE = Gauge("ml_realized_mape", "MAPE จากเฉลยจริง", ["week"])
ESTIMATED_MAPE = Gauge("ml_estimated_mape", "MAPE ที่ NannyML ประเมินโดยไม่ใช้เฉลย", ["week"])
REALIZED_AUC = Gauge("ml_realized_roc_auc", "ROC AUC จากเฉลยจริง (ถ้ามี)", ["week"])
ESTIMATED_AUC = Gauge("ml_estimated_roc_auc", "ROC AUC ที่ NannyML ประเมิน (ถ้ามี)", ["week"])
MULTIVAR = Gauge("ml_multivariate_drift", "ค่า reconstruction error จาก NannyML", ["week"])
LAST_RUN = Gauge("ml_monitoring_last_run_timestamp", "เวลาที่อ่านผลล่าสุด (unix time)")


def refresh() -> None:
    ev_path, nml_path = REPORTS / "evidently_summary.json", REPORTS / "nannyml_summary.json"
    if ev_path.exists():
        for row in json.loads(ev_path.read_text()):
            week = str(row.get("week", "current"))
            DRIFT_SHARE.labels(week).set(row.get("drift_share", 0.0))
            DRIFTED_N.labels(week).set(len(row.get("drifted_features", [])))
            if "mape" in row:
                REALIZED_MAPE.labels(week).set(row["mape"])
            if "roc_auc" in row:
                REALIZED_AUC.labels(week).set(row["roc_auc"])
    if nml_path.exists():
        for row in json.loads(nml_path.read_text()):
            week = str(row.get("week", "current"))
            if "estimated_mape" in row:
                ESTIMATED_MAPE.labels(week).set(row["estimated_mape"])
            if "estimated_roc_auc" in row:
                ESTIMATED_AUC.labels(week).set(row["estimated_roc_auc"])
            MULTIVAR.labels(week).set(row.get("multivariate_drift", 0.0))
    LAST_RUN.set(time.time())


if __name__ == "__main__":
    start_http_server(PORT)
    print(f"drift exporter listening on :{PORT}/metrics")
    while True:
        refresh()
        time.sleep(REFRESH_SEC)
