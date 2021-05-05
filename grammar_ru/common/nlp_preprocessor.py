from typing import *
from grammar_ru.common import validations
import pandas as pd
from .nlp_analyzer import NlpAnalyzer
from .pipeline import run_pipeline_on_text, run_pipeline_on_dataframe


class NlpPreprocessor:
    def __init__(self, analyzers: List[NlpAnalyzer], required_columns=[]):
        self._analyzers = analyzers
        self._required_columns = required_columns

    def _preprocess_dataframe_inner(self, df: pd.DataFrame):
        raise NotImplementedError()

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        self._preprocess_dataframe_inner(df)
        return df

    def run_pipeline_and_preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocess_dataframe(run_pipeline_on_dataframe(self._analyzers, df))

    def run_pipeline_and_preprocess_text(self, text: List[str]) -> pd.DataFrame:
        df = run_pipeline_on_text(self._analyzers, text)
        return self.preprocess_dataframe(df)

    def validate_input(self, df: pd.DataFrame):
        validations.ensure_df_contains(validations.WordCoordinates + self._required_columns, df)
