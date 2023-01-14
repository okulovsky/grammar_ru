from tg.common.ml.batched_training import DataBundle, IndexedDataBundle
from tg.grammar_ru.common import Separator
from tg.grammar_ru.components.plain_context_builder import PlainContextBuilder

from unittest import TestCase
import pandas as pd

src = pd.DataFrame(dict(
    sentence_id=[0] * 10,
    word_id=list(range(10))
))
bundle = DataBundle(src=src)
index = src.iloc[[1, 5, 8]]
index.index.name = 'sample_id'
ibundle = IndexedDataBundle(index, bundle)


class PlainContextBuilderTestCase(TestCase):
    def test_both_sides_with_zero(self):
        bc = PlainContextBuilder(True, 0.5)
        rdf = bc.build_context(ibundle, 5)
        rdf = rdf.reset_index().sort_values(['sample_id','offset'])
        self.assertListEqual([0,1,2,3,   3,4,5,6,7,   6,7,8,9], list(rdf.another_word_id))


    def test_both_sides_wihout_zero(self):
        bc = PlainContextBuilder(False, 0.5)
        rdf = bc.build_context(ibundle, 5)
        rdf = rdf.reset_index().sort_values(['sample_id','offset'])
        self.assertListEqual([0,2,3,    2,3,4,6,7,    5,6,7,9], list(rdf.another_word_id))

    def test_left_with_zero(self):
        bc = PlainContextBuilder(True, 0)
        rdf = bc.build_context(ibundle, 5)
        rdf = rdf.reset_index().sort_values(['sample_id', 'offset'])
        self.assertListEqual([0,1, 1,2,3,4,5, 4,5,6,7,8], list(rdf.another_word_id))


    def test_left_without_zero(self):
        bc = PlainContextBuilder(False, 0)
        rdf = bc.build_context(ibundle, 5)
        rdf = rdf.reset_index().sort_values(['sample_id', 'offset'])
        self.assertListEqual([0,   0,1,2,3,4,   3,4,5,6,7], list(rdf.another_word_id))


    def test_right_with_zero(self):
        bc = PlainContextBuilder(True, 1)
        rdf = bc.build_context(ibundle, 5)
        rdf = rdf.reset_index().sort_values(['sample_id', 'offset'])
        self.assertListEqual([1,2,3,4,5,   5,6,7,8,9,   8, 9], list(rdf.another_word_id))


    def test_right_without_zero(self):
        bc = PlainContextBuilder(False, 1)
        rdf = bc.build_context(ibundle, 5)
        rdf = rdf.reset_index().sort_values(['sample_id', 'offset'])
        self.assertListEqual([2,3,4,5,6,    6,7,8,9,   9], list(rdf.another_word_id))


    def test_odd_proportion(self):
        bc = PlainContextBuilder(True,0.25)
        rdf = bc.build_context(ibundle.change_index(ibundle.index_frame.iloc[[1]]), 5)
        rdf = rdf.reset_index().sort_values(['sample_id', 'offset'])
        self.assertListEqual([4,5,6,7,8], list(rdf.another_word_id))
