from grammar_ru.common.nlp_analyzer import NlpAnalyzer
from grammar_ru.common.natasha import create_chunks_from_dataframe
import pandas as pd


class NatashaMorphAnalyzer(NlpAnalyzer):
    def __init__(self):
        super(NatashaMorphAnalyzer, self).__init__(["word_id"])

    def _analyze_inner(self, df: pd.DataFrame) -> pd.DataFrame:
        chunks = create_chunks_from_dataframe(df)
        pass
