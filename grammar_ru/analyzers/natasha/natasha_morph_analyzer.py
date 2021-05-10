from grammar_ru.common.architecture.nlp_analyzer import NlpAnalyzer
from grammar_ru.common.natasha import create_chunks_from_dataframe
import pandas as pd


class NatashaMorphAnalyzer(NlpAnalyzer):
    def __init__(self):
        super(NatashaMorphAnalyzer, self).__init__()

    def _analyze_inner(self, df: pd.DataFrame):
        chunks = create_chunks_from_dataframe(df)
        pass
