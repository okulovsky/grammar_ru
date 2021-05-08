from typing import *
from tg.common.ml.dft.architecture import DataFrameColumnsTransformer
from grammar_ru.common import validations
import pandas as pd


class NlpPreprocessor(DataFrameColumnsTransformer):
    def __init__(self, required_columns=[]):
        super(DataFrameColumnsTransformer, self).__init__()
        self._required_columns = required_columns

    def fit(self, df: pd.DataFrame):
        # Can be overriden
        return None

    def get_columns(self):
        return self._required_columns

    def _preprocess_dataframe_inner(self, df: pd.DataFrame) -> Iterable[Union[pd.DataFrame, pd.Series]]:
        raise NotImplementedError()

    def transform(self, df: pd.DataFrame):
        self.validate_input(df)
        return self._preprocess_dataframe_inner(df)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        self.validate_input(df)
        for df in self._preprocess_dataframe_inner(df):
            for column in df.columns:
                result[column] = df[column]
        return result

    def validate_input(self, df: pd.DataFrame):
        validations.ensure_df_contains(validations.WordCoordinates + self._required_columns, df)
