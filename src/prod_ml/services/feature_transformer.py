from prod_ml.core.interfaces.feature_transformer import FeatureTransformer
import pandas as pd



class FeatureService(FeatureTransformer):
    """
    Executes feature transformation pipeline.
    Business feature logic stays in domain transformers.
    """

    def __init__(
        self,
        transformers: list[FeatureTransformer]
    ):
        self.transformers = transformers
        self.fitted=False


    def fit(self, data: pd.DataFrame) -> None:
        self.fitted = True


        for transformer in self.transformers:
            transformer.fit(data)




    def transform(self, data:pd.DataFrame) -> pd.DataFrame:

        transformed_data = data

        for transformer in self.transformers:
            transformed_data = transformer.transform(
                transformed_data
            )

        return transformed_data




    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = data
        for transformer in self.transformers:
            transformer.fit(transformed_data)
            transformed_data = transformer.transform(transformed_data)
        return transformed_data