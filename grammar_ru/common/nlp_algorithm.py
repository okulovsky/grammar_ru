from typing import *
import pandas as pd
from grammar_ru.common import validations
from ..dataset_pipeline.nlp_analyzer import NlpAnalyzer
from ..dataset_pipeline.pipeline import run_pipeline


class NlpAlgorithm:
    def __init__(self, analyzers: List[NlpAnalyzer], status_column: str, suggest_column: Optional[str], additional_columns=[]):
        self.analyzers = analyzers
        self._status_column = status_column
        self._suggest_column = suggest_column
        self.additional_columns = additional_columns

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
        validations.ensure_df_contains(['word', 'word_id', 'sentence_id', 'word_index',
                                        'check_requested'] + self.additional_columns, df)

    def run_on_text(self, text: List[str]) -> pd.DataFrame:
        df = run_pipeline(self.analyzers, text)
        df['check_requested'] = True
        self.run(df)
        return df

    def run_on_string(self, s: str) -> pd.DataFrame:
        return self.run_on_text([s])

    def get_name(self):
        return type(self).__name__
