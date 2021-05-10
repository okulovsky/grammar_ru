from typing import *
import pandas as pd
from . import validations
from .separator import Separator


class NlpAlgorithm:
    def __init__(self, status_column: str, suggest_column: Optional[str], required_columns=[]):
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
        validations.ensure_df_contains(
            validations.WordCoordinates + ['check_requested'] + self._required_columns, df)

    def run_on_text(self, text: List[str], paragraphs_to_check=None) -> pd.DataFrame:
        df = Separator.separate_paragraphs(text)
        if paragraphs_to_check is None:
            df['check_requested'] = True
        else:
            df['check_requested'] = df.paragraph_id.isin(paragraphs_to_check)
        self.run(df)
        return df

    def run_on_string(self, s: str, paragraphs_to_check=None) -> pd.DataFrame:
        # TODO: [s] should be replaced with Separator.separate_string, but tests fail.
        return self.run_on_text([s], paragraphs_to_check)

    def get_name(self):
        return type(self).__name__
