from typing import *
from .separator import Separator
import pandas as pd
from grammar_ru.common import validations


class NlpAlgorithm:
    def __init__(self, status_column: str, suggest_column: Optional[str]):
        self._status_column = status_column
        self._suggest_column = suggest_column

    def _run_inner(self, df: pd.DataFrame):
        raise NotImplementedError()

    def run(self, df: pd.DataFrame):
        self.validate_input()
        self._run_inner(df)

    def get_status_column(self):
        return self._status_column

    def get_suggest_column(self):
        return self._suggest_column

    def validate_input(self, df: pd.DataFrame):
        validations.ensure_df_contains(['word', 'word_id', 'sentence_id', 'word_index', 'check_requested'], df)

    def run_on_string(self, s: str):
        df = Separator.separate_string(s)
        df['check_requested'] = True
        self.run(df)
        return df

    def get_name(self):
        return type(self).__name__
