from typing import *
import math
import dataclasses

from pathlib import Path

import pandas as pd
from yo_fluq_ds import *

from .sentence_filterer import SentenceFilterer
from .negative_sampler import NegativeSampler
from ...grammar_ru.ml.features import Featurizer
from ...grammar_ru.ml.corpus import CorpusReader, BucketBalancer, CorpusBuilder
from .transfusion_selector import AlternativeTaskTransfuseSelector



@dataclasses.dataclass
class BundleConfig():
    corpora: List[Path]
    filterer: SentenceFilterer
    negative_sampler: NegativeSampler
    featurizers: List[Featurizer]
    temp_folder: Path


class BundleBuilder():
    def __init__(self, config: AlternativeTaskBundleConfig):
        self.config = config

    def get_all_frames(self, with_progress_bar = True):
        query = CorpusReader.read_frames_from_several_corpora(self.config.corpora)
        if with_progress_bar:
            query = query.feed(fluq.with_progress_bar())
        return query

    def compute_buckets(self):
        en = self.get_all_frames().select(self.config.filterer.filter)
        buckets_df = BucketBalancer.collect_buckets(en)
        return buckets_df

    def prepare(self, buckets: pd.DataFrame, words_per_frame = 50000, words_limit=None):
        balancer = BucketBalancer(BucketBalancer.buckets_statistics_to_dict(buckets))

        selector = AlternativeTaskTransfuseSelector(
            balancer,
            self.config.filterer,
            self.config.negative_sampler
        )

        CorpusBuilder.transfuse_corpus(
            sources=self.config.corpora,
            destination=self.config.temp_folder/'transfuzed.zip',
            selector=selector,
            words_limit=words_limit,
            words_per_frame=words_per_frame,
            overwrite=True
        )

    def featurize(self):
        CorpusBuilder.featurize_corpus(
            self.config.temp_folder/'transfuzed.zip',
            self.config.temp_folder/'featurized.zip',
            self.config.featurizers
        )

    def assemble(self,entries_limit, output_path):
        CorpusBuilder.assemble(
            self.config.temp_folder/'featurized.zip',
            output_path,
            entries_limit
        )