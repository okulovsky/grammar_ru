import dataclasses
import click

from research.common import build_bundle
from tg.grammar_ru.common import Loc
from tg.grammar_ru.ml.tasks.train_index_builder.sentence_filterer import ChtobyFilterer
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import ChtobyIndexBuilder


@dataclasses.dataclass
class DataPath:
    INDEX_PATH = Loc.bundles_path/'chtoby/prepare/raw/raw.zip'
    FEATURES_PATH = Loc.bundles_path/'chtoby/prepare/feat/feat.zip'
    BALANCED_CORPUS_PATH = Loc.bundles_path/'chtoby/prepare/balanced/balanced.zip'
    BUCKET_PATH = Loc.bundles_path/'chtoby/prepare/bucket/bucket.parquet'
    FILTERED_CORPUSES = [
        Loc.bundles_path/'chtoby/prepare/filtered/filtered_lenta.zip',
        Loc.bundles_path/'chtoby/prepare/filtered/filtered_proza.zip'
    ]
    CORPUSES = [
        Loc.corpus_path/'lenta.base.zip',
        Loc.corpus_path/'proza.base.zip',
    ]


@click.group()
def cli() -> None:
    pass


@cli.command()
def balance() -> None:
    bucket_numbers = [2, 3, 4]
    bucket_limit = 2400
    build_bundle.balance(
        DataPath.FILTERED_CORPUSES,
        DataPath.BUCKET_PATH,
        DataPath.BALANCED_CORPUS_PATH,
        bucket_numbers, bucket_limit
    )


@cli.command()
def filter() -> None:
    print('Filtering corpuses')
    filterer = ChtobyFilterer()
    build_bundle.filter_corpuses(filterer, DataPath.CORPUSES)


@cli.command()
def index() -> None:
    print(f'Building index in {DataPath.INDEX_PATH}')
    index_builder = ChtobyIndexBuilder()
    build_bundle.build_index(index_builder, DataPath.BALANCED_CORPUS_PATH, DataPath.INDEX_PATH)


@cli.command()
def features() -> None:
    print(f'Extracting features in {DataPath.FEATURES_PATH}')
    build_bundle.featurize_index(DataPath.INDEX_PATH, DataPath.FEATURES_PATH, workers=3)


@cli.command()
def bundle() -> None:
    bundle_name = 'toy'
    bundle_path = Loc.bundles_path/f'chtoby/{bundle_name}'
    print(f'Building bundle in {bundle_path}')
    build_bundle.assemble(50, DataPath.FEATURES_PATH, bundle_path)


if __name__ == '__main__':
    cli()
