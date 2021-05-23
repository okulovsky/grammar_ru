from grammar_ru.common.architecture.nlp_analyzer import NlpAnalyzer
from .natasha_analyzer import NatashaAnalyzer
from grammar_ru.common.natasha import create_chunks_from_dataframe
from typing import *
import pandas as pd
from functools import reduce


class CombinedNatashaAnalyzer(NlpAnalyzer):
    def __init__(self, analyzers: List[NatashaAnalyzer]):
        super(NlpAnalyzer, self).__init__()
        self._analyzers = analyzers
        self._required_columns = []

    def _analyze_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        chunks = create_chunks_from_dataframe(df)
        results = []

        for analyzer in self._analyzers:
            results.append(analyzer.analyze_chunks(df, chunks))

        return reduce(lambda df1, df2: pd.merge(df1, df2, on='word_id'), results)
