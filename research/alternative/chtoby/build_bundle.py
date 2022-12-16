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
    FILTERED_CORPUS_PATH = Loc.bundles_path/'chtoby/prepare/filtered/filtered.zip'
    BALANCED_CORPUS_PATH = Loc.bundles_path/'chtoby/prepare/balanced/balanced.zip'
    BUCKET_PATH = Loc.bundles_path/'chtoby/prepare/bucket/bucket.zip'


@click.group()
def cli() -> None:
    pass


@cli.command()
def filter() -> None:
    print(f'Building filtered corpus in {DataPath.FILTERED_CORPUS_PATH}')
    filterer = ChtobyFilterer()
    build_bundle.filter_corpuses(filterer, build_bundle.CORPUSES, DataPath.FILTERED_CORPUS_PATH)


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
