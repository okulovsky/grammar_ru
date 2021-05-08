from grammar_ru.common.nlp_analyzer import NlpAnalyzer
from tg.common.datasets.featurization import FeaturizationJob
from tg.common.datasets.access import DataSource
from typing import *
import pandas as pd


class NlpAnalyzationPipeline:
    def __init__(self, analyzers: List[(str, NlpAnalyzer)]):
        self._analyzers = analyzers

    def df_to_featurization_job(self, df: pd.DataFrame, **kwargs) -> FeaturizationJob:
        pass

    def source_to_featurization_job(self, source: DataSource, **kwargs) -> FeaturizationJob:
        pass
