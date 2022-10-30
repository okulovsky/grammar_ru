from tg.grammar_ru.ml.corpus import CorpusBuilder
from tg.grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxStatsFeaturizer, SyntaxTreeFeaturizer
from pathlib import Path
from tg.grammar_ru import Loc

if __name__ == '__main__':
    MDSTORAGE = Loc.processed_path / "proza"
    corpus_path = Path(__file__).parent / 'corpus'

    CorpusBuilder.convert_interformat_folder_to_corpus(
        corpus_path / 'proza.base.zip',
        MDSTORAGE,
        '',
        ['volume']
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
