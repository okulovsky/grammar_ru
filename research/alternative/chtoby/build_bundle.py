from research.common import build_bundle
from tg.grammar_ru.common import Loc
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import ChtobyIndexBuilder


index_path = Loc.bundles_path/'chtoby/prepare/raw/raw.zip'
if not (index_path).exists():
    print('Building index')
    index_builder = ChtobyIndexBuilder()
    build_bundle.build_index(index_builder, build_bundle.LENTA_CORPUS_PATH, index_path)


features_path = Loc.bundles_path/'chtoby/prepare/feat/feat.zip'
if not (features_path).exists():
    print('Extracting features')
    build_bundle.featurize_index(index_path, features_path, workers=3)


bundle_name = 'big'
bundle_path = Loc.bundles_path/f'chtoby/{bundle_name}'
if not (bundle_path).exists():
    print('Building bundle')
    build_bundle.assemble(200, features_path, bundle_path)
