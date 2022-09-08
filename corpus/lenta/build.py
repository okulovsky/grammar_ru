from yo_fluq_ds import *
from tg.grammar_ru.common import Loc
from tg.grammar_ru.ml.corpus import CorpusBuilder, CorpusReader
from tg.grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer, SyntaxTreeFeaturizer
import pandas as pd
from io import BytesIO
import re


def step_0():
    CorpusBuilder.convert_interformat_folder_to_corpus(
        Loc.corpus_path/'lenta.base.zip',
        Loc.processed_path/'lenta',
        '',
        ['volume']
    )

def step_1():
    CorpusBuilder.featurize_corpus(
        Loc.corpus_path/'lenta.base.zip',
        Loc.corpus_path/'lenta.enriched.zip',
        [
            PyMorphyFeaturizer(),
            SlovnetFeaturizer(),
            SyntaxTreeFeaturizer(add_closures=False)
        ]
    )


def check(name):
    reader = CorpusReader(Loc.corpus_path/name)
    toc = reader.get_toc()
    pd.options.display.max_columns = None
    pd.options.display.width = None
    print(len(toc.filename.unique()), toc.shape[0])
    print(toc.filename)
    files = Query.folder(Loc.processed_path/'lenta').select(lambda z: z.name).to_list()
    not_seen = set(files)-set(toc.filename)
    if len(not_seen)>0:
        print(not_seen)
    else:
        print('All files seen')
    print(toc.character_count.sum()//1000000)
    bundle = reader.get_bundles().first()
    print(list(bundle.data_frames))

if __name__ == '__main__':
    pass
    #step_0();check('lenta.base.zip')
    #step_1(); check('lenta.enriched.zip')



