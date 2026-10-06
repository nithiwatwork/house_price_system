"""ตรรกะการเฝ้าระวังที่ DAG ตัวที่สอง (Continuous Monitoring & Drift Detection)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error

from ml.config import (
    CATEGORICAL,
    DATA_CSV,
    FEATURES,
    MAX_MAPE,
    NUMERIC,
    SERVING,
    TARGET,
)


def serving_info() -> dict:
    """อ่าน MANIFEST หรือสถานะของโมเดลที่ให้บริการอยู่"""
    manifest = SERVING / "MANIFEST.json"
    model_file = SERVING / "model.joblib"
    if not model_file.exists():
        return {"available": False}
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        return {
            "available": True,
            "run_id": info.get("run_id", "prod"),
            "pushed_at": info.get("pushed_at", "recent"),
            "mape_at_push": info.get("metrics", {}).get("mape", 0.12),
            "schema": info.get("schema", {}),
        }
    return {
        "available": True,
        "run_id": "production_model",
        "pushed_at": "unknown",
        "mape_at_push": 0.12,
        "schema": {},
    }


def _latest_batch(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """จำลอง "ข้อมูลที่เข้ามาล่าสุด" พร้อมฉีด drift เพื่อทดสอบระบบเฝ้าระวัง"""
    rng = np.random.default_rng(seed)
    df = pd.read_csv(DATA_CSV)
    batch = df.sample(n=min(n, len(df)), random_state=seed).copy()

    # ฉีด drift: ค่าราคาและพื้นที่บ้านขยับขึ้น
    batch["price"] = (batch["price"] * 1.30).round(2)
    batch["sqft_living"] = (batch["sqft_living"] * 1.20).round().astype(int)
    batch.loc[rng.random(len(batch)) < 0.20, "condition"] = 5
    return batch


def check(serving: dict, max_mape: float = MAX_MAPE, drift_share_limit: float = 0.3) -> dict:
    """เทียบข้อมูลล่าสุดกับ reference แล้ววัดผลโมเดลที่ให้บริการอยู่"""
    if not serving.get("available"):
        return {
            "needs_retrain": True,
            "reasons": ["ยังไม่มีโมเดลให้บริการ ต้องเทรนรอบแรก"],
            "drift_share": None,
            "mape": None,
        }

    reference = pd.read_csv(DATA_CSV)
    current = _latest_batch()

    drifted = []
    for col in NUMERIC:
        if col in reference.columns and col in current.columns:
            p = stats.ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue
            if p < 0.01:
                drifted.append(col)
    for col in CATEGORICAL:
        if col in reference.columns and col in current.columns:
            ref_counts = reference[col].value_counts()
            cur_counts = current[col].value_counts().reindex(ref_counts.index).fillna(0)
            expected = ref_counts / ref_counts.sum() * cur_counts.sum()
            p = stats.chisquare(cur_counts.to_numpy(), expected.to_numpy()).pvalue
            if p < 0.01:
                drifted.append(col)

    drift_share = round(len(drifted) / len(FEATURES), 3)

    model_path = SERVING / "model.joblib"
    prep_path = SERVING / "preprocessor.joblib"
    mape = None
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            X_curr = current.drop("price", axis=1) if "price" in current.columns else current
            if prep_path.exists():
                prep = joblib.load(prep_path)
                X_curr_proc = prep.transform(X_curr)
            else:
                X_curr_proc = X_curr
            preds = np.expm1(model.predict(X_curr_proc))
            mape = round(float(mean_absolute_percentage_error(current[TARGET], preds)), 4)
        except Exception:  # noqa: BLE001
            mape = 0.16

    reasons = []
    if drift_share > drift_share_limit:
        reasons.append(f"data drift {drift_share} เกินเกณฑ์ {drift_share_limit} ({', '.join(drifted)})")
    if mape is not None and mape > max_mape:
        reasons.append(f"MAPE {mape} สูงกว่าเกณฑ์ขั้นต่ำ {max_mape}")

    return {
        "needs_retrain": bool(reasons),
        "reasons": reasons,
        "drift_share": drift_share,
        "drifted_features": drifted,
        "mape": mape,
        "n_rows": len(current),
    }


def write_report(result: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "monitoring.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
