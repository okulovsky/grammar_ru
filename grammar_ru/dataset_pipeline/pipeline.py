from typing import *
from grammar_ru.common.separator import Separator
from .nlp_analyzer import NlpAnalyzer
import pandas as pd


def run_pipeline(analyzers: List[NlpAnalyzer], text: List[str]) -> pd.DataFrame:
    result_df = Separator.separate_paragraphs(text)

    for analyzer in analyzers:
        result_df = analyzer.apply(result_df)

    return result_df
