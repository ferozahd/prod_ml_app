from pydantic import BaseModel, Field

class PredictionHeartDiseaseSchema(BaseModel):
    colestorel: int =Field(
        lt=0,
        gt=500,
        description="Colestorel value should be in between 0 and 500"
    )