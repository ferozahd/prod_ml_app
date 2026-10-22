# domains/fraud_detection/models/lifecycle/train.py
from prod_ml.domains.heart_disease.models.factory import ModelFactory


def run_training(X_train, y_train, X_val, y_val, config: dict):
    # 1. Instantiate model via factory based on YAML config
    model = ModelFactory.create(model_type=config["model"]["type"], params=config["model"]["params"])

    # 2. Execute training
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # 3. Log model artifacts and parameters
    return model