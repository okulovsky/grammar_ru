import typing as tp
from pathlib import Path

import pandas as pd
import yo_fluq_ds as yfds

from tg.grammar_ru.common import Loc

from tg.grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxTreeFeaturizer, SyntaxStatsFeaturizer
from tg.grammar_ru.ml.corpus import CorpusReader, CorpusBuilder

from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import DictionaryIndexBuilder
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import NNnIndexBuilder


LENTA_CORPUS_PATH = Loc.corpus_path/'lenta.base.zip'


def read_data(corpus_path: Path) -> tp.Iterable[pd.DataFrame]:
    return (CorpusReader(corpus_path)
            .get_frames()
            .feed(yfds.fluq.with_progress_bar()))


def build_index(
        index_builder: DictionaryIndexBuilder,
        corpus_path: Path,
        bundle_path: Path,
        word_limit: tp.Optional[int] = None
        ) -> None:
    CorpusBuilder.transfuse_corpus(
        [corpus_path],
        bundle_path,
        words_limit=word_limit,
        selector=index_builder.build_train_index
    )


def featurize_index(source: Path, destination: Path, workers: int = 2) -> None:
    CorpusBuilder.featurize_corpus(
        source,
        destination,
        [
            PyMorphyFeaturizer(),
            SlovnetFeaturizer(),
            SyntaxTreeFeaturizer(),
            SyntaxStatsFeaturizer()
        ],
        workers,
        True,
    )


def assemble(
        limit: int,
        corpus_path: Path,
        bundle_path: Path
        ) -> None:
    CorpusBuilder.assemble(
        corpus_path,
        bundle_path,
        limit
    )
    src = pd.read_parquet(bundle_path/'src.parquet')
    index = NNnIndexBuilder.build_index_from_src(src)
    index.to_parquet(bundle_path/'index.parquet')
    print(index.groupby('split').size())
