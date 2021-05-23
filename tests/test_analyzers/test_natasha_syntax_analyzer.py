from grammar_ru.analyzers.natasha.natasha_syntax_analyzer import NatashaSyntaxAnalyzer
from grammar_ru.common.architecture.separator import Separator
from grammar_ru.common.natasha import create_chunks_from_dataframe
from grammar_ru.common.architecture.validations import ensure_df_contains, WordCoordinates
from unittest import TestCase
import pandas as pd
import numpy as np

text = 'Она была красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'


class NatashaSyntaxAnalyzerTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(NatashaSyntaxAnalyzerTestCase, cls).setUpClass()
        cls.syntax = NatashaSyntaxAnalyzer()
        df = Separator.separate_string(text)
        chunks = create_chunks_from_dataframe(df)
        cls.result = cls.syntax.analyze_chunks(df, chunks)
        print(cls.result)

    def test_syntax_general(self):
        self.assertTrue(np.isnan(self.result.loc[(self.result['sentence_id'] == 1) &
                                                 (self.result['word_index'] == 1)]["parent_id"].item()))
        self.assertTrue(np.isnan(self.result.loc[(self.result['sentence_id'] == 1) &
                                                 (self.result['word_index'] == 1)]["rel"].item()))
        self.assertEqual(self.result.loc[(self.result['sentence_id'] == 1) &
                                         (self.result['word_index'] == 0)]["parent_id"].item(), 5)
        self.assertEqual(self.result.loc[(self.result['sentence_id'] == 1) &
                                         (self.result['word_index'] == 0)]["rel"].item(), "nsubj")

    def test_syntax_preserve_coordinates(self):
        try:
            ensure_df_contains(WordCoordinates, self.result)
        except BaseException:
            self.fail("Morph analyzer does not preserve coordinates")
