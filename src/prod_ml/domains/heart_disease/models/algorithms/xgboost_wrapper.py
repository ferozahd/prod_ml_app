# domains/fraud_detection/models/algorithms/xgboost_wrapper.py
from prod_ml.core.base.base_model import BaseModel
import xgboost as xgb

class FraudXGBoostModel(BaseModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def fit(self, X, y, eval_set=None):
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]