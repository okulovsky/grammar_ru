from research.common.bundle_builder import BundleConfig, BundleBuilder
from tg.grammar_ru.common import Loc
from tg.grammar_ru.ml.tasks.train_index_builder.sentence_filterer import ChtobyFilterer
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import ChtobyIndexBuilder
from tg.grammar_ru.ml.features import (
    PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxTreeFeaturizer, SyntaxStatsFeaturizer
)


ChtobyBundleConfig = BundleConfig(
    INDEX_PATH=Loc.bundles_path/'chtoby/prepare/raw/raw.zip',
    FEATURES_PATH=Loc.bundles_path/'chtoby/prepare/feat/feat.zip',
    BALANCED_CORPUS_PATH=Loc.bundles_path/'chtoby/prepare/balanced/balanced.zip',
    BUCKET_PATH=Loc.bundles_path/'chtoby/prepare/bucket/bucket.parquet',
    FILTERED_CORPUSES=[
        Loc.bundles_path/'chtoby/prepare/filtered/filtered_lenta.zip',
        Loc.bundles_path/'chtoby/prepare/filtered/filtered_proza.zip'
    ],
    CORPUSES=[
        Loc.corpus_path/'lenta.base.zip',
        Loc.corpus_path/'proza.base.zip',
    ],
    BUCKETS_NUMBERS=[2, 3, 4],
    BUCKET_LIMIT=2400,
    BUNDLE_NAME='toy',
    TASK_NAME='chtoby',
    SENTENCE_FILTERER=ChtobyFilterer(),
    INDEX_BUILDER=ChtobyIndexBuilder(),
    BUNDLE_LIMIT=50,
    FEATURIZERS=[
        PyMorphyFeaturizer(),
        SlovnetFeaturizer(),
        SyntaxTreeFeaturizer(),
        SyntaxStatsFeaturizer()
    ]
)


class ChtobyBundleBuilder(BundleBuilder):
    def __init__(self) -> None:
        super().__init__(ChtobyBundleConfig)


if __name__ == '__main__':
    builder = ChtobyBundleBuilder()
    builder.features()
