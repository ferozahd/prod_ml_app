from pydantic import BaseModel, Field
from datetime import datetime

class ConfigHeartDiseaseSchema(BaseModel):


    threshold: float = Field(
        ge=0,
        le=1,
        description="Threshold should be in between 0 and 1"
    )
