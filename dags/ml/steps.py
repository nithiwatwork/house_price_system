"""ตรรกะของแต่ละขั้นในไปป์ไลน์ — เขียนเป็นฟังก์ชัน Python ธรรมดา ไม่ import airflow เลย

ทำไมต้องแยกแบบนี้
  1. ทดสอบได้ด้วย pytest ธรรมดาโดยไม่ต้องยก Airflow ขึ้นมา
  2. ย้ายไป orchestrator ตัวอื่น (Kubeflow, Prefect) ได้โดยไม่ต้องเขียนตรรกะใหม่
  3. ไฟล์ DAG จะเหลือแค่ "ลำดับงาน" ซึ่งอ่านง่ายและรีวิวง่าย

ชื่อฟังก์ชันตั้งตามคอมโพเนนต์ของ TFX ใน Lecture 11 เพื่อให้เทียบกันได้ตรง ๆ

ชุดข้อมูล: King County House Sales — ทำนายราคาบ้าน (House Price Regression)
เป้าหมายคือ y = price (USD)
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    BASELINE_MAPE,
    CATEGORICAL,
    DATA_CSV,
    FEATURES,
    MAX_MAPE,
    MAX_MODEL_SIZE_MB,
    MIN_R2,
    NUMERIC,
    SERVING,
    TARGET,
    run_dir,
)


def _safe_copy(src: Path | str, dst: Path | str) -> None:
    """Safe copy that works on Windows / WSL2 Docker bind mounts without failing on utime/copystat."""
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)


def read_house_data(csv_path: Path) -> pd.DataFrame:
    """อ่านไฟล์ข้อมูลราคาบ้านแล้วทำความสะอาดให้เป็นชุดที่ใช้เทรนได้"""
    df = pd.read_csv(csv_path)
    # id และ date ไม่ใช่ฟีเจอร์เชิงกายภาพของตัวบ้าน ตัดออกเพื่อป้องกันสัญญาณรบกวน
    cols_to_drop = [c for c in ["id", "date"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    num_cols = [c for c in NUMERIC if c in df.columns and df[c].isna().any()]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = [c for c in CATEGORICAL if c in df.columns and df[c].isna().any()]
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    return df


# ---------------------------------------------------------------- 1. ExampleGen
def example_gen(run_id: str, data_file: str | None = None) -> dict[str, Any]:
    """อ่านข้อมูลดิบ ทำความสะอาด แล้วแบ่ง train/eval (80/20)

    คืนค่าเป็น dict ของ "เส้นทางไฟล์" ไม่ใช่ DataFrame
    เพราะค่าที่ task ส่งต่อกันใน Airflow จะถูกเก็บใน XCom ซึ่งอยู่ในฐานข้อมูล metadata
    ถ้ายัด DataFrame ลงไปจะทำให้ฐานข้อมูลบวมและ pipeline ช้าลงมาก
    เขียนเป็นไฟล์ แล้วส่งต่อแค่ path
    """
    d = run_dir(run_id)
    if not data_file:
        csv_path = DATA_CSV
    else:
        # ค้นหาตำแหน่งไฟล์ข้อมูลตามโฟลเดอร์ที่เป็นไปได้
        candidates = [
            Path(data_file),
            DATA_CSV.parent / data_file,
            DATA_CSV.parent.parent / "raw" / data_file,
            DATA_CSV.parent.parent / "corrupted" / data_file,
            DATA_CSV.parent.parent / data_file,
        ]
        csv_path = next((p for p in candidates if p.exists() and p.is_file()), DATA_CSV)

    df = read_house_data(csv_path)

    train_df, eval_df = train_test_split(df, test_size=0.20, random_state=42)
    train_path, eval_path = d / "train.csv", d / "eval.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    return {
        "source": str(csv_path),
        "train": str(train_path),
        "eval": str(eval_path),
        "train_path": str(train_path),
        "test_path": str(eval_path),
        "run_dir": str(d),
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "num_rows": len(df),
        "num_cols": len(df.columns),
    }


# ---------------------------------------------------------------- 2. StatisticsGen
def statistics_gen(run_id: str, examples: dict[str, Any]) -> str:
    """สรุปสถิติของชุด train ไว้เป็นไฟล์ JSON ให้ขั้นถัดไปใช้ตรวจ"""
    d = run_dir(run_id)
    train_file = examples.get("train", examples.get("train_path"))
    df = pd.read_csv(train_file)

    stats = {
        "n_rows": len(df),
        "numeric": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": round(float(df[c].mean()), 4),
                "missing": int(df[c].isna().sum()),
            }
            for c in NUMERIC
            if c in df.columns
        },
        "categorical": {
            c: sorted(df[c].dropna().astype(str).unique().tolist())
            for c in CATEGORICAL
            if c in df.columns
        },
        "target_mean": round(float(df[TARGET].mean()), 4) if TARGET in df.columns else None,
    }
    path = d / "statistics.json"
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------- 3. SchemaGen
def schema_gen(run_id: str, statistics_path: str) -> str:
    """สร้าง schema จากสถิติ ถ้ามี schema กลางอยู่แล้วให้ใช้ตัวเดิม

    schema คือ "สัญญา" ของข้อมูล ต้องอยู่นิ่งข้ามการรัน ไม่ใช่สร้างใหม่ทุกครั้ง
    ไม่งั้นข้อมูลเสียจะกลายเป็นมาตรฐานใหม่โดยอัตโนมัติ และ ExampleValidator จะไม่มีวันจับอะไรได้เลย
    """
    from ml.config import INCLUDE

    shared = INCLUDE / "schema.json"
    if shared.exists():
        return str(shared)

    stats = json.loads(Path(statistics_path).read_text(encoding="utf-8"))
    schema = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numeric": {
            c: {
                "min": stats["numeric"][c]["min"],
                "max": stats["numeric"][c]["max"],
                "required": True,
            }
            for c in NUMERIC
            if c in stats.get("numeric", {})
        },
        "categorical": {
            c: {
                "domain": stats["categorical"][c],
                "required": True,
            }
            for c in CATEGORICAL
            if c in stats.get("categorical", {})
        },
        "target": {
            "name": TARGET,
            "min": 1.0,
            "required": True,
        },
    }
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(shared)


# ---------------------------------------------------------------- 4. ExampleValidator
def example_validator(examples: dict[str, Any], schema_path: str) -> dict[str, Any]:
    """ตรวจข้อมูลกับ schema แล้วคืนรายการความผิดปกติ

    ขั้นนี้คือด่านที่ต้อง "หยุด pipeline" เมื่อเจอข้อมูลเสีย
    ถ้าปล่อยผ่านไป เราจะได้โมเดลที่เทรนจากขยะโดยไม่มีใครรู้จนกว่าจะสายเกินไป
    """
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    train_file = examples.get("train", examples.get("train_path"))
    df = pd.read_csv(train_file)
    anomalies: list[str] = []

    # ตรวจสอบตัวแปรเชิงตัวเลข
    for col, rule in schema.get("numeric", {}).items():
        if col not in df.columns:
            anomalies.append(f"{col}: ขาดคอลัมน์ที่ schema กำหนด")
            continue
        if df[col].isna().any():
            anomalies.append(f"{col}: มีค่าว่าง {int(df[col].isna().sum())} แถว")
        lo, hi = float(df[col].min()), float(df[col].max())
        # กฎเชิงโดเมน: คอลัมน์ที่ไม่เคยติดลบ ห้ามติดลบเด็ดขาด ไม่ต้องเผื่อขอบเขต
        if rule.get("min", 0) >= 0 and lo < 0:
            anomalies.append(f"{col}: พบค่าติดลบ {lo:.2f} ทั้งที่ไม่ควรติดลบ")
        # เผื่อขอบเขตไว้ 20% เพราะข้อมูลใหม่ย่อมมีค่าที่กว้างกว่าเดิมได้บ้าง
        span = (rule["max"] - rule["min"]) or 1.0
        if lo < rule["min"] - 0.2 * span or hi > rule["max"] + 0.2 * span:
            anomalies.append(f"{col}: ค่าอยู่นอกช่วงที่คาด ({lo:.2f} ถึง {hi:.2f})")

    # ตรวจสอบเป้าหมายราคา (Target price)
    if TARGET in df.columns:
        invalid_price = int((df[TARGET] <= 0).sum())
        if invalid_price > 0:
            anomalies.append(f"{TARGET}: พบราคา <= 0 จำนวน {invalid_price} แถว ทั้งที่ไม่ควรเกิดขึ้น")

    # ตรวจสอบตัวแปรหมวดหมู่
    for col, rule in schema.get("categorical", {}).items():
        if col not in df.columns:
            anomalies.append(f"{col}: ขาดคอลัมน์ที่ schema กำหนด")
            continue
        unseen = set(df[col].dropna().astype(str).unique()) - {str(x) for x in rule["domain"]}
        if unseen:
            anomalies.append(f"{col}: พบค่าที่ไม่เคยเห็นใน schema {sorted(unseen)[:5]}")

    return {"ok": not anomalies, "anomalies": anomalies}


# ---------------------------------------------------------------- 5. Transform
def transform(run_id: str, examples: dict[str, Any]) -> str:
    """fit ตัวแปลงข้อมูลจากชุด train เท่านั้น แล้วบันทึกเป็น artifact

    ตัวแปลงต้อง fit จาก train เท่านั้น และถูกใช้ซ้ำทั้งตอน eval และตอนให้บริการ
    นี่คือกลไกกัน Training-Serving Skew ที่ Lecture 4 พูดถึง
    """
    d = run_dir(run_id)
    train_file = examples.get("train", examples.get("train_path"))
    train_df = pd.read_csv(train_file)

    num_features = [c for c in NUMERIC if c in train_df.columns]
    cat_features = [c for c in CATEGORICAL if c in train_df.columns]
    valid_features = num_features + cat_features

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    pre.fit(train_df[valid_features])

    path = d / "transform.joblib"
    joblib.dump(pre, path)
    return str(path)


# ---------------------------------------------------------------- 5b. DataCard (task ใหม่ของกลุ่ม)
def data_card_gen(run_id: str, examples: dict[str, Any]) -> str:
    """สร้าง "การ์ดข้อมูล" ของชุด train — ทำงานขนานกับ trainer ได้เพราะไม่พึ่งโมเดล

    task ใหม่ของกลุ่ม: สรุปโปรไฟล์ของข้อมูลไว้อ้างอิงกับผู้มีส่วนได้ส่วนเสีย
    เช่น จำนวนแถว ค่าเฉลี่ยราคา คอลัมน์ที่มีค่าว่าง และจำนวนค่าที่ไม่ซ้ำต่อฟีเจอร์
    """
    d = run_dir(run_id)
    train_file = examples.get("train", examples.get("train_path"))
    df = pd.read_csv(train_file)

    card = {
        "dataset_name": "King County House Sales",
        "n_rows": len(df),
        "n_missing": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
        "n_unique": {c: int(df[c].nunique()) for c in FEATURES if c in df.columns},
        "target_summary": {
            "mean": round(float(df[TARGET].mean()), 2) if TARGET in df.columns else None,
            "median": round(float(df[TARGET].median()), 2) if TARGET in df.columns else None,
            "min": round(float(df[TARGET].min()), 2) if TARGET in df.columns else None,
            "max": round(float(df[TARGET].max()), 2) if TARGET in df.columns else None,
        },
    }
    path = d / "data_card.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------- 6. Trainer
def trainer(run_id: str, examples: dict[str, Any], transform_path: str, max_iter: int = 200) -> str:
    """เทรนโมเดล แล้วบันทึกเป็น Pipeline ก้อนเดียวที่มีตัวแปลงข้อมูลรวมอยู่ด้วย"""
    d = run_dir(run_id)
    train_file = examples.get("train", examples.get("train_path"))
    train_df = pd.read_csv(train_file)
    pre = joblib.load(transform_path)

    valid_features = [c for c in FEATURES if c in train_df.columns]
    X_train = train_df[valid_features]
    y_train = train_df[TARGET].values

    model = Pipeline([
        ("pre", pre),
        (
            "reg",
            HistGradientBoostingRegressor(
                max_iter=int(max_iter),
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
            ),
        ),
    ])
    # เทรนบน log1p(y) เพื่อลด percentage error (MAPE)
    model.fit(X_train, np.log1p(y_train))

    path = d / "model.joblib"
    joblib.dump(model, path)
    return str(path)


# ---------------------------------------------------------------- 7. Evaluator
def evaluator(run_id: str, examples: dict[str, Any], model_path: str) -> dict[str, Any]:
    """วัดผลโมเดลใหม่ เทียบกับเกณฑ์ขั้นต่ำ กับโมเดลที่ให้บริการอยู่ (Baseline)

    คืนค่า blessed = True ก็ต่อเมื่อผ่านทั้งสามเงื่อนไข
    การเทียบกับโมเดลเดิมสำคัญมาก เพราะกันไม่ให้ pipeline เปลี่ยนโมเดลไปเรื่อย ๆ
    ทั้งที่ตัวใหม่ไม่ได้ดีกว่าเดิมจริง
    """
    d = run_dir(run_id)
    eval_file = examples.get("eval", examples.get("test_path"))
    ev = pd.read_csv(eval_file)
    model = joblib.load(model_path)

    valid_features = [c for c in FEATURES if c in ev.columns]
    preds_log = model.predict(ev[valid_features])
    preds = np.expm1(preds_log)
    y_true = ev[TARGET].values

    mape = float(mean_absolute_percentage_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = float(r2_score(y_true, preds))
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    baseline = _current_serving_metrics()
    baseline_mape = baseline.get("mape", BASELINE_MAPE)
    improvement = round(baseline_mape - mape, 4)

    # 3 เงื่อนไข Blessing Gate:
    # 1. MAPE ผ่านเกณฑ์ขั้นต่ำ (< 15%)
    # 2. R2 ผ่านเกณฑ์ขั้นต่ำ (>= 0.70)
    # 3. ขนาดโมเดลต้องไม่เกินเกณฑ์ (< 20 MB) และมีประสิทธิภาพดีกว่าหรือเทียบเท่าเดิม
    passes_threshold = mape < MAX_MAPE and r2 >= MIN_R2
    beats_baseline = improvement >= 0.0 or baseline.get("mape") is None
    passes_size = size_mb < MAX_MODEL_SIZE_MB

    blessed = bool(passes_threshold and beats_baseline and passes_size)

    metrics = {
        "mape": round(mape, 4),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "size_mb": round(size_mb, 2),
        "baseline_mape": baseline_mape,
        "improvement": improvement,
        "passes_threshold": passes_threshold,
        "beats_baseline": beats_baseline,
        "blessed": blessed,
        "n_eval": len(ev),
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def _current_serving_metrics() -> dict[str, Any]:
    manifest = SERVING / "MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("metrics", {})
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- 8. Pusher
def pusher(run_id: str, model_path: str, metrics: dict[str, Any], schema_path: str) -> str:
    """คัดลอกโมเดลที่ผ่านด่านไปยังที่ให้บริการ พร้อมเขียน MANIFEST บันทึกที่มา

    MANIFEST คือบันทึกสายพันธุ์ (lineage) ที่ตอบได้ว่าโมเดลที่ให้บริการอยู่ตอนนี้
    มาจากการรันไหน ใช้ schema ไหน และมีผลวัดเท่าไร ซึ่งเป็นสิ่งที่ผู้ตรวจสอบจะถามหา
    """
    SERVING.mkdir(parents=True, exist_ok=True)
    _safe_copy(model_path, SERVING / "model.joblib")

    manifest = {
        "run_id": run_id,
        "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_model": str(model_path),
        "schema": str(schema_path),
        "metrics": metrics,
    }
    path = SERVING / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
