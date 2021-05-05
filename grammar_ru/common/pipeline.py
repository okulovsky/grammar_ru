from typing import *
from grammar_ru.common.separator import Separator
from .nlp_analyzer import NlpAnalyzer
import pandas as pd


def run_pipeline_on_dataframe(analyzers: List[NlpAnalyzer], df: pd.DataFrame) -> pd.DataFrame:
    for analyzer in analyzers:
        df = analyzer.apply(df)

    return df


def run_pipeline_on_text(analyzers: List[NlpAnalyzer], text: List[str]) -> pd.DataFrame:
    parsed_text_df = Separator.separate_paragraphs(text)

    return run_pipeline_on_dataframe(analyzers, parsed_text_df)
