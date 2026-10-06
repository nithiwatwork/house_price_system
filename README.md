# CP413008 Machine Learning Engineering for Production
# Project: House Price Prediction & MLOps Pipeline

Python 3.11 / 3.12 · Apache Airflow · MLflow · FastAPI · Docker Desktop (WSL 2)
คำสั่งทั้งหมดในคู่มือนี้รันบน Windows (PowerShell) หรือ Docker Desktop

รันตามลำดับหัวข้อ 0 ถึง 7 จะเข้าใจการทำงานครบทุกส่วนของโปรเจกต์

---

## 0. เตรียมตัวก่อนเริ่ม

โปรเจกต์นี้ใช้ชุดข้อมูลราคาบ้าน King County (`kc_house_data.csv`) ในการสร้างระบบทำนายราคาและจัดการ Pipeline แบบครบวงจร

ตรวจดูว่ามีไฟล์ข้อมูลอยู่ที่:
```
data/raw/kc_house_data.csv
```

เปิด **PowerShell** ในโฟลเดอร์โปรเจกต์ แล้วสร้าง Conda Environment สำหรับรันบนเครื่อง:
```powershell
conda create -n project-house python=3.11 -y
conda activate project-house
pip install -r requirements.txt
```

---

## 1. ลำดับการรัน ML Pipeline ด้วยมือ (Local Step-by-Step)

สคริปต์ในโฟลเดอร์ `scripts/` เรียงลำดับขั้นตอน 01 ถึง 04 ไว้อย่างชัดเจน ให้รันตามลำดับนี้:

### ขั้นที่ 1 — ตรวจสอบความถูกต้องของข้อมูล (Data Validation)

```powershell
python scripts/01_data_validation.py
```

หน้าที่: ตรวจสอบข้อมูลดิบเบื้องต้นว่ามีแถวครบตามเกณฑ์ (>= 20,000 แถว), ไม่มี Missing Values และราคาบ้านต้องมากกว่า 0 พร้อมบันทึกผลลง MLflow ในเครื่อง (`mlflow.db`)

สิ่งที่ต้องเห็น:
```
Data loaded successfully from D:\house_price_system\data\raw\kc_house_data.csv
Dataset shape: 21613 rows, 21 columns
Missing values: 0
Price range: $75,000.00 - $7,700,000.00
Validation status: Success
Data validation run finished successfully.
```

ถ้าข้อมูลไม่ผ่านเกณฑ์ สคริปต์จะตัดจบด้วย exit code 1 ทันที เพื่อป้องกันไม่ให้ข้อมูลเสียหลุดไปขั้นตอนถัดไป

### ขั้นที่ 2 — เตรียมข้อมูลและแบ่ง Train / Test (Data Preprocessing)

```powershell
python scripts/02_data_preprocessing.py
```

หน้าที่: แบ่งข้อมูลเป็นชุด Train 80% และ Test 20% แล้วเซฟเป็นไฟล์ `train.csv` กับ `test.csv` ในโฟลเดอร์ `processed_data/` และอัปโหลดไฟล์เก็บเป็น Artifact ใน MLflow

สิ่งที่ต้องเห็น:
```
Starting data preprocessing on D:\house_price_system\data\raw\kc_house_data.csv...
Saved processed data to 'D:\house_price_system\processed_data' directory:
  - train.csv: 17,290 rows
  - test.csv: 4,323 rows
Data preprocessing run finished. Preprocessing Run ID: local_preprocessing_run
```

จำค่า `local_preprocessing_run` หรือ Run ID ที่ได้ไว้ เพื่อนำไปใช้เป็น input ในขั้นถัดไป

### ขั้นที่ 3 — เทรน ประเมินผล และลงทะเบียนโมเดล (Train, Evaluate & Model Registry)

```powershell
python scripts/03_train_evaluate_register.py local_preprocessing_run
```

หน้าที่: นำข้อมูล Train จากข้อ 2 มาสร้าง Pipeline โมเดล `HistGradientBoostingRegressor` (แปลงฟีเจอร์, แปลง Log สเกลราคา) คำนวณค่าชี้วัดความแม่นยำ (MAPE, RMSE, R2)

ก่อนจะยอมรับโมเดล จะมีด่านตรวจคุณภาพโมเดล (Model Quality Gate):
1. ค่าความคลาดเคลื่อน MAPE ต้องน้อยกว่า 15% (`MAPE < 0.15`)
2. ขนาดไฟล์โมเดลต้องไม่เกิน 20 MB

ถ้าผ่านทั้งสองเกณฑ์ โมเดลจะถูกลงทะเบียนใน **MLflow Model Registry** พร้อมติดป้ายกำกับ Alias `@staging` และบันทึกไฟล์ไว้ที่ `artifacts/serving_model/`

สิ่งที่ต้องเห็น:
```
Evaluating Model Quality Gate...
Quality Gate Passed: MAPE=0.12.. (< 0.15) and Size=...MB (< 20.0MB)
Registered model in MLflow with alias '@staging': house_price_model
Training run finished successfully.
```

### ขั้นที่ 4 — ทดสอบโหลดโมเดลมาทำนายผล (Inference / Prediction)

```powershell
python scripts/04_load_and_predict.py
```

หน้าที่: โหลดโมเดลจาก MLflow ที่ติดป้าย `@staging` (หรือดึงจากไฟล์ในเครื่องแบบ fallback) แล้วทดสอบทำนายราคาบ้านตัวอย่าง 1 หลัง

สิ่งที่ต้องเห็น:
```
Loading registered model from MLflow: models:/house_price_model@staging
============================================================
           PREDICTION RESULT (04_load_and_predict)            
============================================================
Property Specs: 3 bed, 2.0 bath, 1800 sqft, Grade 7
Valuation:      $412,345.67 USD
```

---

## 2. สถานการณ์ทดสอบความปลอดภัย — ข้อมูลเสีย ระบบต้องหยุดและแจ้งเตือน

หัวใจสำคัญของ MLOps คือ "ถ้าข้อมูลผิดปกติ ท่อส่งข้อมูลต้องหยุดทันที ไม่ปล่อยให้เทรนต่อ"

### สร้างไฟล์ข้อมูลเสียจำลอง
```powershell
python scripts/make_bad_data.py
```
คำสั่งนี้จะสร้างไฟล์ `data/corrupted/kc_house_corrupted.csv` โดยจงใจใส่ข้อมูลผิดปกติ 3 แบบ:
- ราคาบ้านติดลบ (-500,000)
- จำนวนห้องนอน 99 ห้อง
- ค่าพื้นที่ใช้สอยเป็นค่าว่าง (NaN)

### รันด่านตรวจด้วยข้อมูลเสีย
```powershell
python scripts/01_data_validation.py data/corrupted/kc_house_corrupted.csv
```

สิ่งที่ต้องเห็น:
```
Dataset shape: 100 rows, 21 columns
Missing values: 10
Price range: -$500,000.00 - $1,150,000.00
Validation status: Failed
Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป
```
โปรแกรมจะหยุดทำงานทันที (exit code 1) ไม่ส่งข้อมูลไปต่อที่ขั้นตอน Preprocessing หรือ Training

---

## 3. การตรวจมาตรฐานโค้ดและการทดสอบ (Linting & Automated Testing)

ก่อนที่จะ commit หรือเปิด pull request ต้องทดสอบ 2 คำสั่งนี้ให้ผ่าน

### ตรวจคุณภาพโค้ดด้วย Ruff
```powershell
ruff check scripts/ tests/
```
สิ่งที่ต้องเห็น:
```
All checks passed!
```

### รันชุด Unit Tests ทั้งหมด
```powershell
pytest tests/ -v
```
สิ่งที่ต้องเห็น:
```
======================== 18 passed in ...s ========================
```
ระบบจะตรวจทั้ง Health Check ของ API, โครงสร้างโฟลเดอร์ Data, กฎการทำความสะอาดข้อมูล, การแบ่ง Split แบบไม่มีข้อมูลรั่วไหล (Data Leakage) และความถูกต้องของ Model Artifacts

---

## 4. ให้บริการโมเดลผ่าน REST API (FastAPI Serving)

เมื่อโมเดลผ่านเกณฑ์และบันทึกอยู่ใน `artifacts/serving_model/` แล้ว สามารถเปิด Web API ให้ระบบภายนอกเรียกใช้งานได้

### เปิด API Server
```powershell
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

เปิดดูใน Browser:
- หน้าเอกสาร API (Swagger UI): **http://localhost:8000/docs**
- ตรวจสอบสถานะ Server: **http://localhost:8000/health**

### ยิงคำสั่งทดสอบขอราคาทำนาย (ผ่าน PowerShell หรือ cURL)
```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"features": {"bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "grade": 7, "zipcode": "98178"}}'
```

สิ่งที่ต้องได้กลับมา:
```json
{"predicted_price": 412345.67}
```

---

## 5. การเปิดใช้งานเต็มระบบด้วย Docker Compose (Airflow + Prometheus + Grafana)

สำหรับสภาพแวดล้อมจริงแบบ Production ระบบมี `docker-compose.yml` ที่ยกบริการทั้งหมดขึ้นมาพร้อมกัน

### คำสั่งยกระบบ
```powershell
docker compose up -d --build
docker compose ps
```

### พอร์ตและบริการที่เปิดใช้งาน:
- **FastAPI Serving:** `http://localhost:8000/docs` (สำหรับยิงขอผลทำนาย)
- **Airflow Web UI:** `http://localhost:8082` (User: `airflow`, Password: `airflow`)
- **Prometheus:** `http://localhost:9090` (ฐานข้อมูล Metrics)
- **Grafana:** `http://localhost:3000` (User: `admin`, Password: `admin` สำหรับดูกราฟและ Alert)

### DAGs ใน Airflow (`dags/`):
1. **`house_price_training_pipeline`:** รันการเทรนอัตโนมัติทั้งวงจร (Ingest -> Validate -> Train -> Evaluate -> Bless & Register)
2. **`house_price_monitoring_pipeline`:** ตรวจสอบ Data Drift เป็นประจำทุกวัน ถ้าพบว่าข้อมูลเริ่มเบี่ยงเบนเกินเกณฑ์ จะสั่ง Trigger ให้ DAG เทรนทำงานใหม่อัตโนมัติ

---

## 6. ระบบ CI/CD บน GitHub Actions

ระบบตั้งค่า CI อัตโนมัติไว้ที่ `.github/workflows/ci.yml` โดยจะถูกกระตุ้นเมื่อ:
- มีการ **Push** เข้า Branch `main`
- มีการเปิด **Pull Request**
- หรือกดปุ่ม **Run workflow** ด้วยตนเอง (workflow_dispatch)

ด่านตรวจใน CI แบ่งเป็น 3 ด้านหลัก:
1. **คุณภาพโค้ด (Lint):** Job `lint` รัน `ruff check scripts/ tests/` บน Python 3.12
2. **ความถูกต้องของข้อมูล (Data):** Job `test` รันสคริปต์ `01_data_validation.py` และ `pytest tests/test_data.py` บน Python Matrix (3.11 และ 3.12)
3. **เกณฑ์คุณภาพของโมเดล (Model Quality Gate):** รัน `pytest tests/test_model.py` ตรวจสอบ Artifacts ของโมเดล
4. **สรุปรายงานผล (Artifacts):** อัปโหลดผลตรวจ `report-*.xml` เก็บไว้บน GitHub ด้วย `if: always()` แม้จะมี Step ที่ไม่ผ่าน

---

## 7. คำสั่งปิดระบบ

เมื่อทดสอบเสร็จเรียบร้อย ให้ปิดบริการของ Docker:

```powershell
docker compose down        # หยุดการทำงาน แต่ยังคงประวัติและข้อมูลไว้
docker compose down -v     # หยุดการทำงานและล้างข้อมูลใน Volume ออกทั้งหมด
```
