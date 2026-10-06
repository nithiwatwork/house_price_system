from pydantic import BaseModel, Field

# Schema สำหรับรับข้อมูล Input ในการพยากรณ์
class PredictionRequest(BaseModel):
    area_sqft: float = Field(..., example=1200.0, description="ขนาดพื้นที่ (ตร.ฟุต)")
    bedrooms: int = Field(..., example=3, description="จำนวนห้องนอน")
    bathrooms: int = Field(..., example=2, description="จำนวนห้องน้ำ")
    age_years: int = Field(..., example=5, description="อายุของบ้าน (ปี)")

# Schema สำหรับส่งผลลัพธ์ Output กลับไป
class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str