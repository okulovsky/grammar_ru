from tg.grammar_ru.ml.corpus import CorpusBuilder, BucketCorpusBalancer
from tg.grammar_ru.ml.tasks.style import AuthorSelector
from tg.grammar_ru.common import Loc

from tg.grammar_ru.ml.tasks.style.bundle import StyleIndexBuilder

from tg.grammar_ru.ml.features import (
    PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxTreeFeaturizer, SyntaxStatsFeaturizer,
    GloveFeaturizer, DialogMarkupFeaturizer, DummyFeaturizer
)


def select_authors() -> None:
    tranfuse_kwargs = \
    [
        {
            "sources": [Loc.corpus_path/"books.base.zip"],
            "destination": Loc.corpus_path/"books.martin.zip",
            "selector": AuthorSelector(["Мартин"]).select
        },

        {
            "sources": [Loc.corpus_path/"ficbook.base.zip",],
            "destination": Loc.corpus_path/"ficbook.martin.zip",
            "selector": AuthorSelector(["Martin"], author_column_name = "fandom").select
        }
    ]

    for kwargs in tranfuse_kwargs:
        CorpusBuilder.transfuse_corpus(**kwargs)


def build_buckets_frame() -> None:
    BucketCorpusBalancer.build_buckets_frame(
        corpus_list = [
            Loc.corpus_path/"books.base.zip",
            Loc.corpus_path/"proza.base.zip"
        ],
        bucket_path = Loc.corpus_path/"prepare/buckets/books_proza.parquet"
    )

def filter_by_bucket_numbers() -> None:
    BucketCorpusBalancer.filter_buckets_by_bucket_numbers(
        bucket_path = Loc.corpus_path/"prepare/buckets/books_proza.parquet",
        bucket_numbers = [2,3,4],
        destination_path = Loc.corpus_path/"prepare/buckets/books_proza_filtered.parquet"
    )

def balancing() -> None:
    import math
    balancer = BucketCorpusBalancer(
        buckets = Loc.corpus_path/"prepare/buckets/books_proza_filtered.parquet",
        bucket_limit = 45000,
        log_base = math.e
    )

    CorpusBuilder.transfuse_corpus(
        sources = [
            Loc.corpus_path/"books.base.zip",
            Loc.corpus_path/"proza.base.zip"
        ],
        destination = Loc.corpus_path/"prepare/balance/books_proza.zip",
        selector = balancer.select
    )

def build_index() -> None:
    index_builder = StyleIndexBuilder()

    CorpusBuilder.transfuse_corpus(
        sources = [Loc.corpus_path/"prepare/balance/books_proza.zip"],
        destination = Loc.bundles_path/"style/prepare/raw/books_proza.zip",
        selector = index_builder.select
    )

def featurize_index() -> None:
    CorpusBuilder.featurize_corpus(
        source = Loc.bundles_path/"style/prepare/raw/books_proza.zip",
        destination = Loc.bundles_path/"style/prepare/feat/books_proza.zip",
        # steps = [
        #     PyMorphyFeaturizer(),
        #     SlovnetFeaturizer(),
        #     SyntaxTreeFeaturizer(),
        #     SyntaxStatsFeaturizer()
        # ],
        # steps = [
        #     PyMorphyFeaturizer(),
        #     GloveFeaturizer(disable_downloading = True),
        #     DummyFeaturizer(unnecessary_keys = ["pymorphy"])
        # ],
        steps = [
            PyMorphyFeaturizer(),
            SlovnetFeaturizer(),
            SyntaxTreeFeaturizer(),
            SyntaxStatsFeaturizer(),
            GloveFeaturizer(disable_downloading = True)
        ],
        workers = 2,
        append = True,
    )

def assemble(name: str, limit: int) -> None:
    bundle_path = Loc.bundles_path/f"style/{name}"
    CorpusBuilder.assemble(
        corpus_path = Loc.bundles_path/"style/prepare/feat/books_proza.zip",
        bundle_path = bundle_path,
        limit_entries = limit,
        random_state = 42
    )
    import pandas as pd

    src = pd.read_parquet(bundle_path/"src.parquet")
    index = StyleIndexBuilder.build_index_from_src(src)
    index.to_parquet(bundle_path/"index.parquet")
    print(index.groupby("split").size())


if __name__ == "__main__":
    # select_authors()
    build_buckets_frame()
    filter_by_bucket_numbers()
    balancing()
    build_index()
    featurize_index()
    assemble("books_proza_all_3x45k", None)
    pass
