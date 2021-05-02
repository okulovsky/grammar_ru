from typing import *
from .separator import Separator
import pandas as pd

class NlpAlgorithm:
    def __init__(self, status_column: str, suggest_column: Optional[str]):
        self._status_column = status_column
        self._suggest_column = suggest_column

    def run(self, df):
        raise NotImplementedError()

    def get_status_column(self):
        return self._status_column

    def get_suggest_column(self):
        return self._suggest_column

    def validate_input(self, df):
        for column in ['word','word_id','sentence_id','word_index','check_requested']:
            if column not in df.columns:
                raise ValueError(f"Column `{column}` not in dataframe")


    def run_on_string(self, s: str):
        df = Separator.separate_string(s)
        df['check_requested'] = True
        self.run(df)
        return df
    
    def get_name(self):
        return type(self).__name__