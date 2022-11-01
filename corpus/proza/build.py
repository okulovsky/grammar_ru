# -*- coding: utf-8 -*-
from tg.grammar_ru.ml.corpus import CorpusBuilder
from tg.grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxStatsFeaturizer, SyntaxTreeFeaturizer
from pathlib import Path
from tg.grammar_ru import Loc
import importlib
import sys
sys.setdefaultencoding("utf-8")
# # sys.setdefaultencoding() does not exist, here!
# importlib.reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')

if __name__ == '__main__':
    MDSTORAGE = Loc.processed_path / "proza"
    corpus_path = Path(__file__).parent / 'corpus'

    CorpusBuilder.convert_interformat_folder_to_corpus(
        corpus_path / 'proza.base.zip',
        MDSTORAGE,
        '',
        None  # ['volume']
    )

    featurizers = [
        PyMorphyFeaturizer(),
        SlovnetFeaturizer(),
        SyntaxTreeFeaturizer(),
        SyntaxStatsFeaturizer()
    ]

    # CorpusBuilder.featurize_corpus(
    #     corpus_path / 'proza.base.zip',
    #     corpus_path / 'proza.featurized.zip',
    #     featurizers
    # )
