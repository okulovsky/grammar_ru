import math
import dataclasses
import typing as tp
from pathlib import Path

import pandas as pd
from tg.grammar_ru.common import Loc
from tg.grammar_ru.ml.features import Featurizer
from tg.grammar_ru.ml.corpus import CorpusBuilder, BucketCorpusBalancer
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import IndexBuilder
from tg.grammar_ru.ml.tasks.train_index_builder.sentence_filterer import SentenceFilterer


@dataclasses.dataclass
class BundleConfig():
    INDEX_PATH: Path
    FEATURES_PATH: Path
    BALANCED_CORPUS_PATH: Path
    BUCKET_PATH: Path
    FILTERED_CORPUSES: tp.List[Path]
    CORPUSES: tp.List[Path]

    BUCKETS_NUMBERS: tp.List[int]
    BUCKET_LIMIT: int

    BUNDLE_NAME: str
    TASK_NAME: str

    SENTENCE_FILTERER: SentenceFilterer
    INDEX_BUILDER: IndexBuilder

    BUNDLE_LIMIT: int

    FEATURIZERS: tp.List[Featurizer]


class BundleBuilder():
    def __init__(self, config: BundleConfig):
        self.config = config

    def balance(self) -> None:
        BucketCorpusBalancer.build_buckets_frame(self.config.FILTERED_CORPUSES, self.config.BUCKET_PATH)

        balancer = BucketCorpusBalancer(
            buckets=pd.read_parquet(self.config.BUCKET_PATH),
            log_base=math.e,
            bucket_limit=self.config.BUCKET_LIMIT,
        )

        CorpusBuilder.transfuse_corpus(
            sources=self.config.FILTERED_CORPUSES,
            destination=self.config.BALANCED_CORPUS_PATH,
            selector=balancer
        )

    def filter(self) -> None:
        for corpus_path in self.config.CORPUSES:
            corpus_name = corpus_path.name.split('.')[0]
            filtered_path = Loc.bundles_path/f'{self.config.TASK_NAME}/prepare/filtered/filtered_{corpus_name}.zip'

            CorpusBuilder.transfuse_corpus(
                [corpus_path],
                filtered_path,
                selector=self.config.SENTENCE_FILTERER
            )

    def index(self) -> None:
        CorpusBuilder.transfuse_corpus(
            [self.config.BALANCED_CORPUS_PATH],
            self.config.INDEX_PATH,
            words_limit=None,
            selector=self.config.INDEX_BUILDER
        )

    def features(self) -> None:
        CorpusBuilder.featurize_corpus(
            self.config.INDEX_PATH,
            self.config.FEATURES_PATH,
            self.config.FEATURIZERS,
            3,
            True,
        )

    def bundle(self) -> None:
        bundle_path = Loc.bundles_path/f'{self.config.TASK_NAME}/{self.config.BUNDLE_NAME}'

        CorpusBuilder.assemble(
            self.config.FEATURES_PATH,
            bundle_path,
            self.config.BUNDLE_LIMIT
        )
        src = pd.read_parquet(bundle_path/'src.parquet')
        index = IndexBuilder.build_index_from_src(src)
        index.to_parquet(bundle_path/'index.parquet')
