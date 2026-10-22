# domains/fraud_detection/schemas/raw_features_schema.py
#Validates incoming data before ingestion using tools like Pydantic, Pandera, or Great Expectations.
from pydantic import BaseModel, Field
from datetime import datetime
# domains/heart_disease/schemas/raw_patient.py

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class RawPatientEvent(BaseModel):
    patient_id: str

    age: int = Field(
        ge=0,
        le=120,
        description="Patient age"
    )

    sex: str = Field(
        description="Male/Female"
    )

    chest_pain_type: str

    resting_bp: float = Field(
        gt=0,
        description="Resting blood pressure"
    )

    cholesterol: float = Field(
        ge=0,
        description="Serum cholesterol level"
    )

    fasting_bs: int = Field(
        ge=0,
        le=1,
        description="Fasting blood sugar indicator"
    )

    resting_ecg: str

    max_hr: float = Field(
        gt=0,
        description="Maximum heart rate achieved"
    )

    exercise_angina: str

    old_peak: float = Field(
        ge=0,
        description="ST depression induced by exercise"
    )

    st_slope: str

    timestamp: datetime

    device_id: Optional[str] = None