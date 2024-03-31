import unittest
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from tg.common import DataBundle
from tg.common.ml import batched_training as bt
from grammar_ru.corpus import CorpusBuilder, CorpusReader
from grammar_ru.common import Loc, Separator
from grammar_ru.features import PyMorphyFeaturizer, SlovnetFeaturizer
from tg.projects.punct.context_builder import PunctContextBuilder


@dataclass
class CONTEXT_TEST_CASE:
    sentence: str
    is_target: tp.Sequence[bool]
    ground_truth: tp.Sequence[int]


TEST_CASES = {
    'COMMA_AT_THE_END': CONTEXT_TEST_CASE(
        'Это предложение заканчивается запятой,',
        [False, False, False, True, False],
        [0, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, -2, -1, 0]
    ),
    'COMMA_AT_THE_BEGINNING': CONTEXT_TEST_CASE(
        ', Это предложение начинается с запятой.',
        [False, False, False, False, False, False, False],
        [0, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0]
    ),
    'COMMA_IN_THE_MIDDLE': CONTEXT_TEST_CASE(
        'Запятая стоит где-то, в центре предложения',
        [False, False, True, False, False, False, False],
        [0, 1, -1, 0, 1, -2, -1, 0, 2, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0]
    ),
    'TWO_COMMAS_IN_CONTEXT': CONTEXT_TEST_CASE(
        'В предложении, внезапно, две запятые',
        [False, True, False, True, False, False, False],
        [0, 1, -1, 0, 2, -2, -1, 0, 1, -2, -1, 0, 2, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0]
    ),
}


class PunctContextBuilderTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_to_right_contexts_proportion = 0.5
        self.context_size = 5
        self.builder = PunctContextBuilder(
            include_zero_offset=True,
            left_to_right_contexts_proportion=self.left_to_right_contexts_proportion
        )

    def run_test(self, test_case: CONTEXT_TEST_CASE):
        idb = self.build_idb_from_test_case(test_case)
        context = self.builder.build_context(idb, self.context_size)
        offsets = context.sort_index().index.get_level_values(1).values

        self.assertListEqual(list(offsets), list(test_case.ground_truth))

    def test_comma_at_the_end(self):
        self.run_test(TEST_CASES['COMMA_AT_THE_END'])

    def test_comma_at_the_beginning(self):
        self.run_test(TEST_CASES['COMMA_AT_THE_BEGINNING'])

    def test_comma_in_the_middle(self):
        self.run_test(TEST_CASES['COMMA_IN_THE_MIDDLE'])

    def test_two_commas_in_context(self):
        self.run_test(TEST_CASES['TWO_COMMAS_IN_CONTEXT'])

    def build_idb_from_test_case(self, test_case: CONTEXT_TEST_CASE):
        src = Separator.separate_string(test_case.sentence)
        src['is_target'] = test_case.is_target
        index_frame = src[['word_id', 'sentence_id']]
        index_frame.index = index_frame.index.rename('sample_id')

        db = DataBundle(src=src)
        idb = bt.IndexedDataBundle(index_frame, db)

        return idb
