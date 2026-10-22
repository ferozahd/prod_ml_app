import numpy as np
import pandas as pd
from pandas import DataFrame

from prod_ml.core.interfaces.feature_transformer import FeatureTransformer


class HeartCompleteFeatureTransformation(FeatureTransformer):

    def fit(self, data: DataFrame) -> None:
        pass

    def transform(self, data:DataFrame) -> DataFrame:
        data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y': 1, 'N': 0}).astype(int)

        data['Age_MaxHR_Ratio'] = data['MaxHR'] / data['Age']

        data['Stress_Index'] = data['Oldpeak'] * data['ExerciseAngina']

        data['Age_Group'] = pd.cut(
            data['Age'],
            bins=[0, 40, 50, 60, 100],
            labels=['Young', 'Middle', 'Adult', 'Senior']
        )

        data['Cholesterol_missing'] = (data['Cholesterol'] == 0).astype(int)
        data['RestingBP_missing'] = (data['RestingBP'] == 0).astype(int)

        # Replace invalid values with NaN

        data['Cholesterol'] = data['Cholesterol'].replace(0, np.nan)
        data['RestingBP'] = data['RestingBP'].replace(0, np.nan)

        # Fill using median

        data['Cholesterol'] = data['Cholesterol'].fillna(data['Cholesterol'].median())
        data['RestingBP'] = data['RestingBP'].fillna(data['RestingBP'].median())

        return data