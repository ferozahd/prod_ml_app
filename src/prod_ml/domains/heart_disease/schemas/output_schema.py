from pydantic import BaseModel, Field
from datetime import datetime


class OutputHeartDiseaseSchema(BaseModel):

    disease: float = Field(
        ge=0,
        le=1,
        description="Threshold should be in between 0 and 1"
    )
    confident: float = Field(
        ge=0,
        le=1,
        description="Threshold should be in between 0 and 1"
    )