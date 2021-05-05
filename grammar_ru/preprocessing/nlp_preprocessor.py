from typing import *
from grammar_ru.common import validations
import pandas as pd
from ..dataset_pipeline.nlp_analyzer import NlpAnalyzer
from ..dataset_pipeline.pipeline import run_pipeline


class NlpPreprocessor:
    def __init__(self, analyzers: List[NlpAnalyzer], additional_columns=[]):
        self._analyzers = analyzers
        self._additional_columns = additional_columns

    def _preprocess_inner(self, df: pd.DataFrame):
        raise NotImplementedError()

    def preprocess(self, text: List[str]):
        df = run_pipeline(self._analyzers, text)
        self.validate_input(df)
        self._preprocess_inner(df)
        return df

    def validate_input(self, df: pd.DataFrame):
        validations.ensure_df_contains(['word', 'word_id', 'sentence_id', 'word_index'] + self._additional_columns, df)
