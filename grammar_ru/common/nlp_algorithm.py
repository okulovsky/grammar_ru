from typing import *
import pandas as pd
from grammar_ru.common import validations
from .nlp_preprocessor import NlpPreprocessor


class NlpAlgorithm:
    def __init__(self, preprocessor: NlpPreprocessor, status_column: str, suggest_column: Optional[str], required_columns=[]):
        self._preprocessor = preprocessor
        self._status_column = status_column
        self._suggest_column = suggest_column
        self._required_columns = required_columns

    def _run_inner(self, df: pd.DataFrame):
        raise NotImplementedError()

    def run(self, df: pd.DataFrame):
        self.validate_input(df)
        self._run_inner(df)

    def get_status_column(self):
        return self._status_column

    def get_suggest_column(self):
        return self._suggest_column

    def validate_input(self, df: pd.DataFrame):
        validations.ensure_df_contains(validations.WordCoordinates + ['check_requested'] + self._required_columns, df)

    def run_on_text(self, text: List[str]) -> pd.DataFrame:
        df = self._preprocessor.run_pipeline_and_preprocess_text(text)
        df['check_requested'] = True
        self.run(df)
        return df

    def run_on_string(self, s: str) -> pd.DataFrame:
        return self.run_on_text([s])

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df['check_requested'] = True
        self.run(self._preprocessor.run_pipeline_and_preprocess_dataframe(df))
        return df

    def get_name(self):
        return type(self).__name__
