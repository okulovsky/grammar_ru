from grammar_ru.analyzers.natasha.natasha_morph_analyzer import NatashaMorphAnalyzer
from grammar_ru.common.architecture.separator import Separator
from grammar_ru.common.natasha import create_chunks_from_dataframe
from grammar_ru.common.architecture.validations import ensure_df_contains, WordCoordinates
from unittest import TestCase
import pandas as pd

text = 'Она была красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'


class NatashaMorphAnalyzerTestCase(TestCase):
    def setUp(self) -> None:
        self.morph = NatashaMorphAnalyzer()
        df = Separator.separate_string(text)
        chunks = create_chunks_from_dataframe(self.df)
        self.result = self.morph.analyze_chunks(df, chunks)
        print(self.result)

    def test_morph_general(self):
        self.assertEqual(self.result.loc[(self.result['sentence_id'] == 0) &
                                         (self.result['word_index'] == 2)]["POS"].item(), "ADJ")
        self.assertEqual(self.result.loc[(self.result['sentence_id'] == 1) &
                                         (self.result['word_index'] == 1)]["POS"].item(), "VERB")
        self.assertEqual(self.result.loc[(self.result['sentence_id'] == 0) &
                                         (self.result['word_index'] == 0)]["Gender"].item(), "Fem")

    def test_morph_preserve_coordinates(self):
        try:
            ensure_df_contains(WordCoordinates, self.result)
        except BaseException:
            self.fail("Morph analyzer does not preserve coordinates")
    
