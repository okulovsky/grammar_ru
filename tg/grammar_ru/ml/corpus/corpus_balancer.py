import pandas as pd
import numpy as np 
import math

from typing import List, Dict, Union, Optional
from .transfuse_selector import ITransfuseSelector
from .corpus_reader import read_data
from pathlib import Path
import deprecated


def create_path_if_not_exists(path: Path) -> None:
    path_dirs = Path("/".join(path.parts[:-1]))
    if not path_dirs.exists():
        path_dirs.mkdir(parents = True)

@deprecated.deprecated('Use BucketBalances instead')
class BucketCorpusBalancer(ITransfuseSelector):
    def __init__(
        self, buckets: Union[pd.DataFrame, Path], 
        log_base: float, bucket_limit: int = 10000, random_state: int = 42) -> None:

        super(BucketCorpusBalancer, self).__init__()

        if isinstance(buckets, Path):
            buckets = pd.read_parquet(buckets)

        buckets["sentences_sample"] = (
            buckets.sentences.apply(lambda x: pd.Series(x).
            sample(n = min(len(x), bucket_limit), random_state = random_state).to_list())
        )
        self.buckets, self.log_base = buckets, log_base

    @staticmethod
    def extract_bucket_frame(df: pd.DataFrame, log_base: float = math.e) -> pd.DataFrame:
        agg_mapping = {
            "log_len": ("word", lambda x: round(math.log(x.count(), log_base))),
            "corpus": ("corpus_id", lambda x: x.iloc[0])
        }

        df = df.groupby(["sentence_id"]).agg(**agg_mapping)

        df["bucket"] = df.corpus + "/" + df.log_len.astype(str)
        df = (
            df.reset_index().groupby(["bucket"]).
            sentence_id.apply(list).reset_index(name = "sentences")
        )
        df["bucket_size"] = df["sentences"].apply(lambda x: len(x))
        return df.set_index("bucket")

    @staticmethod
    def build_buckets_frame(
        corpus_list: Union[List[Path], Path], bucket_path: Path, log_base: float = math.e) -> None:
        buckets = pd.DataFrame()
        for df in read_data(corpus_list):
            df = BucketCorpusBalancer.extract_bucket_frame(df, log_base)
            buckets = (
                pd.concat([buckets, df]).reset_index().
                groupby(["bucket"]).agg({"sentences": "sum", "bucket_size": "sum"})
            )
        
        create_path_if_not_exists(bucket_path)
        buckets.to_parquet(bucket_path)

    @staticmethod
    def filter_buckets_by_bucket_numbers(
        bucket_path: Path, bucket_numbers: List[int], destination_path: Optional[Path] = None) -> None:
        buckets = pd.read_parquet(bucket_path)
        indices = pd.Index(
            filter(
                lambda x: int(x.split("/")[-1]) in bucket_numbers, 
                buckets.index
            )
        )
        buckets = buckets.loc[indices]
        if not destination_path:
            buckets.to_parquet(bucket_path)
        else:
            create_path_if_not_exists(destination_path)
            buckets.to_parquet(destination_path)

    def select(
        self, corpus: Path, df: pd.DataFrame, toc_row: Dict) -> Union[List[pd.DataFrame], pd.DataFrame]:
        if "original_corpus_id" not in df.columns:
            df['original_corpus_id'] = df.corpus_id

        bdf = BucketCorpusBalancer.extract_bucket_frame(df, self.log_base)
        intersect_indices = self.buckets.index.intersection(bdf.index)
        buckets = self.buckets.loc[intersect_indices]
        bdf = bdf.loc[intersect_indices]

        if not len(buckets): return []

        indices = np.concatenate(list(map(
            lambda p: np.array(list(set(p[0]).intersection(p[1]))),
            zip(buckets.sentences_sample, bdf.sentences)
        )))

        return df[df.sentence_id.isin(indices)]
