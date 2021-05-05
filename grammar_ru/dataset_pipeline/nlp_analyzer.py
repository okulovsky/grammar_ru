from typing import *
from grammar_ru.common import validations
import pandas as pd

class NlpAnalyzer:
    def __init__(self, join_by_columns: List[str], additional_required_columns = []):
        self.required_columns = additional_required_columns
        self.join_by = join_by_columns

    def validate_input(self, df: pd.DataFrame):
        validations.validate_df_contains(['word', 'word_id', 'sentence_id', 'word_index'] + self.required_columns, df)
    
    def _analyze_inner(self, df:  pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def analyze(self, df: pd.DataFrame):
        self.validate_input(df)
        return self._analyze_inner(df)
    
    def apply(self, df: pd.DataFrame):
        return df.join(self.analyze(df), on=self.join_by)