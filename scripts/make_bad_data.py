"""
สร้างไฟล์ข้อมูล "เสีย" ของ King County House Sales ไว้สาธิตว่า pipeline หยุดและแจ้งเตือนได้จริง

ใส่ความผิดปกติสองแบบที่ต่างชนิดกัน:
  1. ค่าหมวดหมู่ที่ไม่เคยเห็นมาก่อน หรือหลุดช่วง เช่น grade=99 หรือ zipcode ผิดปกติ
  2. ค่าตัวเลขที่เป็นไปไม่ได้ เช่น price ติดลบ หรือ bedrooms=99
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "data" / "raw" / "kc_house_data.csv"
dst = ROOT / "data" / "corrupted" / "kc_house_corrupted.csv"


df = pd.read_csv(src)
corrupted_df = df.head(100).copy()

# 1) ค่าตัวเลขที่เป็นไปไม่ได้ — ราคาบ้านติดลบ
corrupted_df.loc[corrupted_df.index[:10], "price"] = -500000.0

# 2) ค่าห้องนอนที่เป็นไปไม่ได้
corrupted_df.loc[corrupted_df.index[10:20], "bedrooms"] = 99

# 3) ข้อมูลสูญหายในฟีเจอร์สำคัญ
corrupted_df.loc[corrupted_df.index[20:30], "sqft_living"] = None

dst.parent.mkdir(parents=True, exist_ok=True)
corrupted_df.to_csv(dst, index=False)
print("เขียนไฟล์ข้อมูลเสียแล้ว:", dst)
print("ความผิดปกติที่ใส่ไว้: price=-500,000 (ราคาติดลบ), bedrooms=99 (ห้องนอนผิดปกติ), sqft_living=NaN (ค่าสูญหาย)")
