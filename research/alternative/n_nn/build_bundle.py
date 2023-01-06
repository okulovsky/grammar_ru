from pathlib import Path

import yo_fluq_ds as yfds

from research.common import build_bundle
from tg.grammar_ru.common import Loc
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import NNnIndexBuilder
from tg.grammar_ru.ml.tasks.n_nn.word_normalizer import RegexNormalizer
from tg.grammar_ru.ml.tasks.n_nn.bundle import build_dictionary


VOCAB_FILE = Path(__file__).parent/'words.json'


if not (VOCAB_FILE).exists():
    print('Collecting words')
    words = list(build_dictionary(
        build_bundle.read_data(build_bundle.LENTA_CORPUS_PATH),
        RegexNormalizer()))
    yfds.FileIO.write_json(words, VOCAB_FILE)


index_path = Loc.bundles_path/'n_nn/prepare/raw/raw.zip'
if not (index_path).exists():
    print('Building index')
    words = yfds.FileIO.read_json(VOCAB_FILE)
    index_builder = NNnIndexBuilder(words, word_normalizer=RegexNormalizer())
    build_bundle.build_index(index_builder, build_bundle.LENTA_CORPUS_PATH, index_path)


features_path = Loc.bundles_path/'n_nn/prepare/feat/feat.zip'
if not (features_path).exists():
    print('Extracting features')
    build_bundle.featurize_index(index_path, features_path, workers=3)


bundle_name = 'big_1'
bundle_path = Loc.bundles_path/f'n_nn/{bundle_name}'
if not (bundle_path).exists():
    print('Building bundle')
    build_bundle.assemble(200, features_path, bundle_path)
