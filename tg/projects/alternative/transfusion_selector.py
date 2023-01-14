from typing import *
from .negative_sampler import NegativeSampler
from .sentence_filterer import SentenceFilterer
from tg.grammar_ru.corpus import ITransfuseSelector
from pathlib import Path
import pandas as pd


class AlternativeTaskTransfuseSelector(ITransfuseSelector):
    def __init__(self,
                 balancing_selector: ITransfuseSelector,
                 sentence_filter: SentenceFilterer,
                 negative_sampler: NegativeSampler
                 ):
        self.balancing_selector = balancing_selector
        self.sentence_filter = sentence_filter
        self.negative_sampler = negative_sampler

    def select(self, corpus: Path, df: pd.DataFrame, toc_row: Dict) -> Union[List[pd.DataFrame], pd.DataFrame]:
        df = self.balancing_selector.select(corpus, df, toc_row)
        if df.shape[0] == 0:
            return []
        df = self.sentence_filter.filter(df)
        if df.shape[0] == 0:
            return []
        negatives = self.negative_sampler.build_all_negative_samples_from_positive(df)
        dfs = [df] + negatives
        for i, df in enumerate(dfs):
            df['label'] = 0 if i==0 else 1
        return dfs





