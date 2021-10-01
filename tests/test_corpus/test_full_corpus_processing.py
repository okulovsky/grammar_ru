from grammar_ru.ml.corpus import CorpusBuilder, CorpusReader
from grammar_ru.common import *
from unittest import TestCase
import shutil
import os
from pathlib import Path
import pandas as pd
from grammar_ru.ml.features import PyMorphyFeaturizer, SlovnetFeaturizer

def filter(db: DataBundle):
    if db.src.shape[0]<400:
        return False
    return True

class CorpusBuilderTestCase(TestCase):
    def test_corpus_building(self):
        path = Loc.temp_path/'tests/corpus_building'
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)

        corpus_1 = 'test_corpus.1.zip'
        builder = CorpusBuilder(path,Path(__file__).parent/'samples')
        builder.convert_interformat_folder_to_corpus(corpus_1, '', ['field1', 'field2'])

        rd = CorpusReader(path/corpus_1)
        pd.options.display.max_columns=None
        pd.options.display.width = None
        toc = rd.get_toc()

        self.assertListEqual([0,1,2,3,0,1,2], list(toc.part_index))
        self.assertListEqual(['test_turing']*4+['singularity']*3, list(toc.field2))

        steps = [
            filter,
            PyMorphyFeaturizer(),
            SlovnetFeaturizer().as_enricher()
        ]
        corpus_2 = 'test_corpus.2.zip'
        builder.enrich_corpus(corpus_1, corpus_2, steps)

        rd = CorpusReader(path/corpus_2)
        toc = rd.get_toc()
        self.assertEqual(3, toc.shape[0])
        db = rd.get_bundles().first()
        self.assertSetEqual({'src','pymorphy','slovnet'}, set(db.data_frames))


