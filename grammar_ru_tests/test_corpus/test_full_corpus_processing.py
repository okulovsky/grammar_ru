from grammar_ru.corpus import CorpusBuilder, CorpusReader
from grammar_ru.common import *
from unittest import TestCase
import shutil
import os
from pathlib import Path
import pandas as pd
from grammar_ru.features import PyMorphyFeaturizer, SlovnetFeaturizer

MAKE_COPY = True

def filter(db: DataBundle):
    if db.src.shape[0]<400:
        return False
    return True

class CorpusBuilderTestCase(TestCase):
    def test_corpus_building(self):
        path = Loc.temp_path/'tests/corpus_building'
        md_folder = Path(__file__).parent / 'samples'
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)

        corpus_1 = path/'test_corpus.1.zip'
        builder = CorpusBuilder()
        builder.convert_interformat_folder_to_corpus(corpus_1, md_folder, ['field1', 'field2'])


        rd = CorpusReader(corpus_1)
        pd.options.display.max_columns=None
        pd.options.display.width = None
        toc = rd.get_toc()
        print(toc)

        self.assertListEqual([0,1,2,0,1,2,3], list(toc.part_index))
        self.assertListEqual(['singularity']*3+['test_turing']*4, list(toc.field2))

        steps = [
            filter,
            PyMorphyFeaturizer(),
            SlovnetFeaturizer()
        ]
        corpus_2 = path/'test_corpus.2.zip'
        builder.featurize_corpus(corpus_1, corpus_2, steps)

        rd = CorpusReader(corpus_2)
        toc = rd.get_toc()
        self.assertEqual(3, toc.shape[0])
        db = rd.get_bundles().first()
        self.assertSetEqual({'src','pymorphy','slovnet'}, set(db.data_frames))

        builder.assemble(
            corpus_2,
            path/'bundle',
        )

        db = DataBundle.load(path/'bundle')
        self.assertEqual(db.src.shape[0], db.src.drop_duplicates('word_id').shape[0])

        if MAKE_COPY:
            shutil.copy(corpus_1, Loc.test_corpus_basic)
            shutil.copy(corpus_2, Loc.test_corpus_enriched)
            shutil.rmtree(Loc.test_bundle)
            shutil.copytree(path/'bundle', Loc.test_bundle)



