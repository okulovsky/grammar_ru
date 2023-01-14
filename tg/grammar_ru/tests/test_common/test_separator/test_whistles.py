from unittest import TestCase
from tg.grammar_ru.common.separator import Separator, _generate_offsets
import pandas as pd

pd.options.display.width=None

class SeparatorTestCase(TestCase):
    def test_reset_indices(self):
        df = Separator.separate_paragraphs(['Первый. Абзац','Второй абзац'])
        df.word_id*=10
        df.sentence_id*=100
        df.paragraph_id*=1000
        ndf = Separator.reset_indices(df, 50)
        print(df)
        self.assertListEqual([50,51,52,53,54], list(ndf.word_id))
        self.assertListEqual([50,50,51,52,52], list(ndf.sentence_id))
        self.assertListEqual([50,50,50,51,51], list(ndf.paragraph_id))
        self.assertListEqual([50,51,52,53,54], list(ndf.index))
        self.assertEqual(56, Separator.get_max_id(ndf))
        self.assertListEqual(list(df.word_id), list(ndf.original_word_id))
        self.assertListEqual(list(df.sentence_id), list(ndf.original_sentence_id))
        self.assertListEqual(list(df.paragraph_id), list(ndf.original_paragraph_id))
        self.assertListEqual([False]*5, list(ndf.updated))
