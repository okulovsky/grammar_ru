from grammar_ru.common.architecture.nlp_analyzer import NlpAnalyzer
from .natasha_analyzer import NatashaAnalyzer
from grammar_ru.common.natasha import create_chunks_from_dataframe
from typing import *
import pandas as pd


class CombinedNatashaAnalyzer(NlpAnalyzer):
    def __init__(self, analyzers: List[NatashaAnalyzer]):
        super(NlpAnalyzer, self).__init__()
        self._analyzers = analyzers

    def _analyze_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        chunks = create_chunks_from_dataframe(df)
        results = {}

        for analyzer in self._analyzers:
            results[analyzer.get_name()] = analyzer.analyze_chunks(df, chunks).add_prefix(analyzer.get_name())

        return pd.DataFrame(results)
