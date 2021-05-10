from typing import *
from tg.common.datasets.featurization.featurizer import DataframeFeaturizer
from grammar_ru.common import validations
import pandas as pd


class NlpAnalyzer(DataframeFeaturizer):
    def __init__(self, required_columns=[]):
        super(DataframeFeaturizer, self).__init__()
        self._required_columns = required_columns

    def validate_input(self, df: pd.DataFrame):
        validations.ensure_df_contains(validations.WordCoordinates + self._required_columns, df)

    def _featurize(self, obj):
        # Keep whole dataframe as it was before.
        return [obj]

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.analyze(df)

    def _analyze_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self._analyze_inner(df)
