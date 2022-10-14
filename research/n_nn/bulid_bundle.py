from pathlib import Path

import yo_fluq_ds as yfds

import research.bundle_common as bundle_common

from tg.grammar_ru.common import Loc

from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import NNnIndexBuilder


VOCAB_FILE = Path(__file__).parent/'words.json'


if not (Path(__file__).parent/'words.json').exists():
    bundle_common.build_word_dict(VOCAB_FILE)


words = yfds.FileIO.read_json(VOCAB_FILE)
index_builder = NNnIndexBuilder(words)
bundle_path = Loc.bundles_path/'n_nn/prepare/raw/raw.zip'
bundle_common.build_index(index_builder, bundle_common.LENTA_CORPUS_PATH, bundle_path)
