from .nlp_analyzer import NlpAnalyzer
from tg.common.datasets.access import DataSource
from typing import *
import pandas as pd
from grammar_ru.common.architecture.separator import Separator


class NlpAnalyzationPipeline:
    def __init__(self, analyzers: List[Tuple[str, NlpAnalyzer]]):
        self._analyzers = analyzers

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        for (name, analyzer) in self._analyzers:
            result = result.merge(analyzer.analyze(df).add_prefix(name + '_'))

        return result

    def analyze(self, source: DataSource) -> pd.DataFrame:
        df = pd.DataFrame(source.get_data())
        return self.analyze_dataframe(df)

    def analyze_text(self, text: List[str]) -> pd.DataFrame:
        parsed_text_df = Separator.separate_paragraphs(text)

        return self.analyze_dataframe(parsed_text_df)

    def analyze_string(self, string: str) -> pd.DataFrame:
        return self.analyze_text(Separator.separate_string(string))
