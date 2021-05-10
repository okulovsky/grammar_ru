from ..common.architecture.nlp_preprocessor import NlpPreprocessor
import pandas as pd


class DummyPreprocessor(NlpPreprocessor):
    def __init__(self):
        super(DummyPreprocessor, self).__init__()

    def _preprocess_dataframe_inner(self, df: pd.DataFrame):
        yield df
