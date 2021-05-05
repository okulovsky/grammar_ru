from grammar_ru.common.nlp_analyzer import NlpAnalyzer
import pandas as pd


class PyMorphAnalyzer(NlpAnalyzer):
    def __init__(self):
        super(PyMorphAnalyzer, self).__init__()

    def _apply_inner(self, df: pd.DataFrame):
        pass
