# Pipeline that combines two things:
# 1) NlpAnalyzationPipeline
# 2) NlpPreprocessingPipeline

# Scenarios:
# 1) I should be able to send NlpAnalizationPipeline to create dataset on cloud
#   because it can require a lot of resources. It means, that NlpAnalyzationPipeline should
#   be either inherited from FeaturizationJob or be able to return it conviniently.
# 2) I should be able to use NlpPreprocessingPipeline to apply preprocessing before passing
#   the data to the algorithm.


# An algorithm should receive NlpPipeline containing both analyzation and preprocessing
# in order to be able to transform new data accordingly.
# The problem here is that we probably want to prevent scenarios when you trained model using one
# pipeline and then applying model using other pipeline. It could be prevented if NlpAlgorithm
# contained some methods for model training, but I think that this architecture would not be
# flexible and convenient enough for testing and experimenting with data.
# So the user should be careful when analyzing and applying its results.

from ..preprocessors.dummy_preprocessor import DummyPreprocessor
from .nlp_analyzation_pipeline import NlpAnalyzationPipeline
from tg.common.ml import dft
from typing import *
import pandas as pd


class NlpPipeline:
    @staticmethod
    def empty():
        return NlpPipeline(NlpAnalyzationPipeline([]), dft.DataFrameTransformer([DummyPreprocessor()]))

    def __init__(self, analyzation_pipeline: NlpAnalyzationPipeline, preprocessing_pipeline: dft.DataFrameTransformer):
        self._analyzation_pipeline = analyzation_pipeline
        self._preprocessing_pipeline = preprocessing_pipeline

    def run_on_text(self, text: List[str]) -> pd.DataFrame:
        analyzed_text = self._analyzation_pipeline.analyze_text(text)
        return self._preprocessing_pipeline.transform(analyzed_text)  # TODO: Data is not fitted. How to handle this?

    def run_on_string(self, string: str) -> pd.DataFrame:
        return self.run_on_text([string])
