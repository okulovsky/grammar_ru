from tg.grammar_ru.ml.tasks.tsa.bundle import build_dictionary
from tg.grammar_ru.ml.tasks.train_index_builder.train_index_builders import TsaIndexBuilder
from tg.grammar_ru.ml.corpus import CorpusReader, CorpusBufferedWriter, CorpusBuilder
from tg.grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxTreeFeaturizer, SyntaxStatsFeaturizer
from tg.grammar_ru.common import Loc
from tg.common import S3Handler
from yo_fluq_ds import *
from pathlib import Path

VOCAB_FILE = Path(__file__).parent/'words.json'

def read_data():
    return CorpusReader(Loc.corpus_path/'lenta.base.zip').get_frames().feed(fluq.with_progress_bar())


def build_word_dict():
    words = list(build_dictionary(read_data()))
    print(words)
    print(len(words))
    FileIO.write_json(words, VOCAB_FILE)

def build_index():
    words = FileIO.read_json(VOCAB_FILE)
    index_builder = TsaIndexBuilder(words)
    CorpusBuilder.transfuse_corpus(
        [Loc.corpus_path/'lenta.base.zip'],
        Loc.bundles_path/'tsa/prepare/raw/raw.zip',
        selector = index_builder.build_train_index
    )

def featurize_index():
    CorpusBuilder.featurize_corpus(
        Loc.bundles_path / 'tsa/prepare/raw/raw.zip',
        Loc.bundles_path / 'tsa/prepare/feat/feat.zip',
        [
            PyMorphyFeaturizer(),
            SlovnetFeaturizer(),
            SyntaxTreeFeaturizer(),
            SyntaxStatsFeaturizer()
        ],
        2,
        True,
    )

def assemble(name, limit):
    bundle_path = Loc.bundles_path/f'tsa/{name}'
    CorpusBuilder.assemble(
        Loc.bundles_path / 'tsa/prepare/feat/feat.zip',
        bundle_path,
        limit
    )
    src = pd.read_parquet(bundle_path/'src.parquet')
    index = TsaIndexBuilder.build_index_from_src(src)
    index.to_parquet(bundle_path/'index.parquet')
    print(index.groupby('split').size())

def upload_bundle(name):
    bundle_path = Loc.bundles_path / f'tsa/{name}'
    S3Handler.upload_folder(
        'ps-data-science-sandbox',
        'sagemaker/tsa/datasets/'+name,
        bundle_path)



if __name__ == '__main__':
    pass
    #build_word_dict()
    #build_index()
    #featurize_index()
    #assemble('toy', 5)
    #assemble('big', 50)
    #assemble('full', None)
    #upload_bundle('big')

