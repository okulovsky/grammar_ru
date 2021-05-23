from grammar_ru.analyzers.natasha.natasha_morph_analyzer import NatashaMorphAnalyzer
from grammar_ru.common.architecture.separator import Separator
from grammar_ru.common.natasha import create_chunks_from_dataframe
from unittest import TestCase
import pandas as pd

text = 'Она была красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'
pd.options.display.max_columns = None
pd.options.display.width = None


class NatashaMorphAnalyzerTestCase(TestCase):
    def setUp(self) -> None:
        self.morph = NatashaMorphAnalyzer()
        self.df = Separator.separate_string(text)
        self.chunks = create_chunks_from_dataframe(self.df)

    def test_simple(self):
        result = self.morph.analyze_chunks(self.df, self.chunks)
        self.assertEqual(result.loc[(result['sentence_id'] == 0) & (result['word_index'] == 2)]["POS"].item(), "ADJ")
