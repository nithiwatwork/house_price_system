from prometheus_client import start_http_server, Gauge
import time
import pandas as pd
from scipy.stats import ks_2samp

# 1. กำหนด Prometheus Metrics
DATA_DRIFT_SCORE = Gauge('data_drift_p_value', 'P-value for feature data drift test')
CONCEPT_DRIFT_SCORE = Gauge('concept_drift_mape', 'Current MAPE score for model drift')

def check_data_drift(reference_data, current_data, feature_col):
    # คำนวณ Data Drift ด้วย KS-test
    stat, p_value = ks_2samp(reference_data[feature_col], current_data[feature_col])
    DATA_DRIFT_SCORE.set(p_value) # อัปเดตค่าไปยัง Prometheus
    return p_value

if __name__ == '__main__':
    # เปิดเซิร์ฟเวอร์ส่ง Metrics ให้ Prometheus ดึง (Scrape) ที่ Port 8000
    start_http_server(8000)
    print("Drift Detector Metrics server running on port 8000...")
    
    # รัน Loop ตรวจสอบ Drift เป็นระยะ
    while True:
        # โหลดข้อมูลจริง ( Reference vs Current Data )
        # p_val = check_data_drift(ref_df, curr_df, 'price')
        time.sleep(30) # รันคำนวณทุกๆ 30 วินาที