from grammar_ru.analyzers.natasha.natasha_morph_analyzer import NatashaMorphAnalyzer
from grammar_ru.analyzers.natasha.natasha_syntax_analyzer import NatashaSyntaxAnalyzer
from grammar_ru.analyzers.natasha.combined_natasha_analyzer import CombinedNatashaAnalyzer
from grammar_ru.common.architecture.separator import Separator
from grammar_ru.common.architecture.validations import ensure_df_contains
from unittest import TestCase
import numpy as np

text = 'Она была красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'


class CombinedNatashaAnalyzerTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(CombinedNatashaAnalyzerTestCase, cls).setUpClass()
        cls.analyzer = CombinedNatashaAnalyzer([NatashaMorphAnalyzer(), NatashaSyntaxAnalyzer()])
        df = Separator.separate_string(text)
        cls.result = cls.analyzer._analyze_inner(df)
        print(cls.result)

    def test_morph_and_syntax_general(self):
        self.assertEqual(self.result.loc[(self.result['word_id'] == 2)]["POS"].item(), "ADJ")
        self.assertEqual(self.result.loc[(self.result['word_id'] == 5)]["POS"].item(), "VERB")
        self.assertEqual(self.result.loc[(self.result['word_id'] == 0)]["Gender"].item(), "Fem")
        self.assertEqual(self.result.loc[(self.result['word_id'] == 5)]["parent_id"].item(), -1)
        self.assertTrue(np.isnan(self.result.loc[(self.result['word_id'] == 5)]["rel"].item()))
        self.assertEqual(self.result.loc[(self.result['word_id'] == 4)]["parent_id"].item(), 5)
        self.assertEqual(self.result.loc[(self.result['word_id'] == 4)]["rel"].item(), "nsubj")

    def test_combined_preserve_coordinates(self):
        try:
            ensure_df_contains(["word_id"], self.result)
        except BaseException:
            self.fail("Combined analyzer does not preserve word_id")
