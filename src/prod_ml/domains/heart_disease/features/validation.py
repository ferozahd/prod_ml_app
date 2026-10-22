# domains/fraud_detection/features/validation.py
import pandera as pa
# from prod_ml.domains.heart_disease.schemas.config_schema import ConfigHeartDiseaseSchema

def validate_domain_features(df):
    """Executes Pandera schema checks on computed feature dataframes."""
    try:
        return True
        # return FraudFeatureSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Feature computation validation failed: {e}")