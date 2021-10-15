from typing import *
import pandas as pd
from ..common import Separator


class NlpAlgorithm:
    def __init__(self, status_column: str, suggest_column: Optional[str], hint_column: Optional[str]):
        self._status_column = status_column
        self._suggest_column = suggest_column
        self._hint_column = hint_column

    def _run_inner(self, df: pd.DataFrame):
        raise NotImplementedError()

    def run(self, df: pd.DataFrame):
        Separator.check_df(df)
        if 'check_requested' not in df.columns:
            df['check_requested'] = True
        self._run_inner(df)


    def get_status_column(self):
        return self._status_column

    def get_suggest_column(self):
        return self._suggest_column

    def get_hint_column(self):
        return self._hint_column


    def put_check_requested(self, df: pd.DataFrame, paragraphs_to_check=None):
        if 'check_requested' in df.columns:
            return
        if paragraphs_to_check is None:
            df['check_requested'] = True
        else:
            df['check_requested'] = df.paragraph_id.isin(paragraphs_to_check)


    def run_on_text(self, text: List[str], paragraphs_to_check=None) -> pd.DataFrame:
        df = Separator.separate_paragraphs(text)
        self.put_check_requested(df, paragraphs_to_check)
        self.run(df)
        return df

    def run_on_string(self, s: str, paragraphs_to_check=None) -> pd.DataFrame:
        df = Separator.separate_string(s)
        self.put_check_requested(df, paragraphs_to_check)
        self.run(df)
        return df

    def get_name(self):
        return type(self).__name__
