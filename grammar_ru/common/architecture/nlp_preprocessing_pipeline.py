from .nlp_preprocessor import NlpPreprocessor
from tg.common.ml import dft
from typing import *

# Can be completely replaced with dft.DataFrameTransformer?


class NlpPreprocessingPipeline:
    def __init__(self, preprocessors: List[NlpPreprocessor]):
        self._preprocessors = preprocessors

    pass
