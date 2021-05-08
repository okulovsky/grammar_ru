from grammar_ru.common.nlp_analyzer import NlpAnalyzer
from grammar_ru.common.natasha import create_chunks_from_dataframe
import pandas as pd


class NatashaSyntaxAnalyzer(NlpAnalyzer):
    def __init__(self):
        super(NatashaSyntaxAnalyzer, self).__init__()

    def _analyze_inner(self, df: pd.DataFrame):
        chunks = create_chunks_from_dataframe(df)
        pass
