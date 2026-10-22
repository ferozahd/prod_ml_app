
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from prod_ml.core.interfaces.feature_transformer import FeatureTransformer

class CategoricalFeatureEncoder(FeatureTransformer):

    CATEGORICAL_COLUMNS = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
        "Age_Group",
    ]

    def __init__(self):
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            drop=None,
            sparse_output=False,
            dtype=int,
        )
        self.feature_names = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        missing = [
            col for col in self.CATEGORICAL_COLUMNS
            if col not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if data[self.CATEGORICAL_COLUMNS].isna().any().any():
            raise ValueError(
                "Missing categorical values. "
                "Run PreprocessorService first."
            )

        self.encoder.fit(data[self.CATEGORICAL_COLUMNS])

        self.feature_names = (
            self.encoder.get_feature_names_out(
                self.CATEGORICAL_COLUMNS
            )
        )
        self.is_fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("CategoricalEncoder is not fitted")

        missing = [
            col for col in self.CATEGORICAL_COLUMNS
            if col not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if data[self.CATEGORICAL_COLUMNS].isna().any().any():
            raise ValueError(
                "Missing categorical values. "
                "Run PreprocessorService first."
            )

        encoded = self.encoder.transform(
            data[self.CATEGORICAL_COLUMNS]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.feature_names,
            index=data.index,
        )

        return pd.concat(
            [
                data.drop(columns=self.CATEGORICAL_COLUMNS),
                encoded_df,
            ],
            axis=1,
        )
