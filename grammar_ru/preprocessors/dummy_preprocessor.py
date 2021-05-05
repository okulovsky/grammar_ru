from ..preprocessing.nlp_preprocessor import NlpPreprocessor
import pandas as pd


class DummyPreprocessor(NlpPreprocessor):
    def __init__(self):
        super(DummyPreprocessor, self).__init__([])

    def _preprocess_inner(self, df: pd.DataFrame):
        pass
