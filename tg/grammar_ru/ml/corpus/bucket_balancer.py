from .transfuse_selector import ITransfuseSelector
import pandas as pd
from pathlib import Path
from typing import *
import numpy as np

class BucketBalancer(ITransfuseSelector):
    def __init__(self, buckets: Dict[str,Iterable[int]]):
        self.buckets = buckets

    def select(self, corpus: Path, df: pd.DataFrame, toc_row: Dict) -> Union[List[pd.DataFrame], pd.DataFrame]:
        corpus_id = df.corpus_id.unique()
        if len(corpus_id)!=1:
            raise ValueError(f'Corpus_id is expected to be unique within dataframe, but there were {corpus_id}')
        corpus_id = corpus_id[0]
        return df.loc[df.sentence_id.isin(self.buckets[corpus_id])]

    @staticmethod
    def collect_buckets(dfs: Iterable[pd.DataFrame], log_base = 2):
        result = []
        for df in dfs:
            sdf = df.groupby(['corpus_id','sentence_id']).size().to_frame('len').reset_index()
            sdf['log_len'] = np.log(sdf.len)/np.log(log_base)
            sdf.log_len = sdf.log_len.astype(int)
            sdf = sdf.drop('len',axis=1)
            result.append(sdf)
        rdf = pd.concat(result)
        rdf = rdf.reset_index(drop=True)
        return rdf

    @staticmethod
    def buckets_statistics_to_dict(df: pd.DataFrame) -> Dict[str, Iterable[int]]:
        s = df.groupby('corpus_id').sentence_id.apply(set)
        return s.to_dict()

