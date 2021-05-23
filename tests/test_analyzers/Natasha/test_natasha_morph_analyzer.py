from grammar_ru.analyzers.natasha.natasha_morph_analyzer import NatashaMorphAnalyzer
from grammar_ru.common.architecture.separator import Separator
from grammar_ru.common.natasha import create_chunks_from_dataframe
from grammar_ru.common.architecture.validations import ensure_df_contains
from unittest import TestCase
import pandas as pd

text = 'Она была красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'


class NatashaMorphAnalyzerTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(NatashaMorphAnalyzerTestCase, cls).setUpClass()
        cls.morph = NatashaMorphAnalyzer()
        df = Separator.separate_string(text)
        chunks = create_chunks_from_dataframe(df)
        cls.result = cls.morph.analyze_chunks(df, chunks)
        print(cls.result)

    def test_morph_general(self):
        self.assertEqual(self.result.loc[(self.result['word_id'] == 2)]["POS"].item(), "ADJ")
        self.assertEqual(self.result.loc[(self.result['word_id'] == 5)]["POS"].item(), "VERB")
        self.assertEqual(self.result.loc[(self.result['word_id'] == 0)]["Gender"].item(), "Fem")

    def test_morph_preserve_coordinates(self):
        try:
            ensure_df_contains(["word_id"], self.result)
        except BaseException:
            self.fail("Morph analyzer does not preserve word_id")
