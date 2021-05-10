from typing import *
from grammar_ru.common import validations
import pandas as pd


class NlpAnalyzer:
    def __init__(self, required_columns=[]):
        self._required_columns = required_columns

    def validate_input(self, df: pd.DataFrame):
        validations.ensure_df_contains(validations.WordCoordinates + self._required_columns, df)

    def _analyze_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self._analyze_inner(df)
