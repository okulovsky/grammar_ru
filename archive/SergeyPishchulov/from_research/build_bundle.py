from research.grammatical_gender.bundle import GGTrainIndexBuilder
from tg.grammar_ru.ml.corpus import CorpusBuilder
from tg.grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxTreeFeaturizer, SyntaxStatsFeaturizer
from tg.grammar_ru.common import Loc
from yo_fluq_ds import *

# VOCAB_FILE = Path(__file__).parent / 'words.json'

#
# def read_data():
#     return CorpusReader(Loc.corpus_path / 'lenta.base.zip').get_frames().feed(fluq.with_progress_bar())


def build_index():
    gg_index_builder = GGTrainIndexBuilder()
    CorpusBuilder.transfuse_corpus(
        [Loc.corpus_path / 'lenta.base100.zip'],
        Loc.bundles_path / 'grammatical_gender/prepare/lenta100/raw/raw.zip',
        selector=gg_index_builder.build_train_index
    )


def featurize_index():
    CorpusBuilder.featurize_corpus(
        Loc.bundles_path / 'grammatical_gender/prepare/lenta100/raw/raw.zip',
        Loc.bundles_path / 'grammatical_gender/prepare/lenta100/feat/feat.zip',
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
    bundle_path = Loc.bundles_path / f'grammatical_gender/{name}'
    CorpusBuilder.assemble(
        Loc.bundles_path / 'grammatical_gender/prepare/lenta100/feat/feat.zip',
        bundle_path,
        limit
    )
    src = pd.read_parquet(bundle_path / 'src.parquet')
    index = GGTrainIndexBuilder.build_index_from_src(src)
    index.to_parquet(bundle_path / 'index.parquet')
    print(index.groupby('split').size())


# def upload_bundle(name):
#     bundle_path = Loc.bundles_path / f'tsa/{name}'
#     S3Handler.upload_folder(
#         'ps-data-science-sandbox',
#         'sagemaker/tsa/datasets/' + name,
#         bundle_path)


if __name__ == '__main__':
    build_index()
    featurize_index()
    # assemble('tiny3featdist', 5)
    assemble('toy3featdist', 5)
    # # assemble('toy3', 5)
    assemble('mid3featdist', 20)
    assemble('big3featdist', 50)
    # assemble('tiny', None)
    # upload_bundle('big')
