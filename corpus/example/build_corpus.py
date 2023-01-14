from tg.grammar_ru.corpus import CorpusBuilder
from tg.grammar_ru.features import PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxStatsFeaturizer, SyntaxTreeFeaturizer
from pathlib import Path

if __name__ == '__main__':
    corpus_path = Path(__file__).parent / 'corpus'

    CorpusBuilder.convert_interformat_folder_to_corpus(
        corpus_path/'example.base.zip',
        Path(__file__).parent/'processed',
        '',
        ['original_file_name']
    )

    featurizers = [
        PyMorphyFeaturizer(),
        SlovnetFeaturizer(),
        SyntaxTreeFeaturizer(),
        SyntaxStatsFeaturizer()
    ]

    CorpusBuilder.featurize_corpus(
        corpus_path/'example.base.zip',
        corpus_path/'example.featurized.zip',
        featurizers
    )