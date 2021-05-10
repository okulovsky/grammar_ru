from grammar_ru.common.architecture.nlp_analyzer import NlpAnalyzer
import pandas as pd
from grammar_ru.common.natasha import create_chunks_from_dataframe
from typing import *


class NatashaAnalyzer(NlpAnalyzer):
    def __init__(self, required_columns=[]):
        super(NlpAnalyzer, self).__init__(required_columns)

    def _analyze_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        chunks = create_chunks_from_dataframe(df)
        return self.analyze_chunks(chunks)

    def analyze_chunks(self, chunks: List[List[str]]) -> pd.DataFrame:
        raise NotImplementedError()
