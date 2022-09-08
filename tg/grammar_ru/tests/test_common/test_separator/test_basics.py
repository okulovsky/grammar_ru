from unittest import TestCase
from tg.grammar_ru.common.separator import Separator, _generate_offsets
import pandas as pd

class SeparatorTestCase(TestCase):
    def test_generate_offsets(self):
        text = '  A BCD   EFG  H '
        tails, length = _generate_offsets(text, ['A','BCD','EFG'])
        self.assertListEqual([1,3, 2], tails)
        self.assertEqual(15, length)

    def test_separation_columns(self):
        text = '«Какой-нибудь»   текст —  с знаками… И еще словами!.. Вот так.'
        df = Separator.separate_string(text)
        self.assertListEqual(Separator.COLUMNS, list(df.columns))

    def test_separation(self):
        text = '«Какой-нибудь»   текст —  с знаками… И еще словами!.. Вот так.'
        df = Separator.separate_string(text)
        self.assertListEqual(list(df.word_tail),        [0, 0,  3, 1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0])
        self.assertListEqual(list(df.word_length), [1, 12, 1, 5, 1, 1, 7, 1, 1, 3, 7, 3, 3, 3, 1])

    def test_separation_multi(self):
        df = Separator.separate_paragraphs(['первое предожение. Второе.', 'Второй параграф'])
        #pd.set_option('max_columns',None); pd.set_option('display.width', 1000);print(df)
        self.assertListEqual([0, 0, 0, 1, 1, 2, 2], list(df.sentence_id))
        self.assertListEqual([0, 1, 2, 3, 4, 5, 6], list(df.word_id))
        self.assertListEqual([0, 0, 0, 0, 0, 1, 1], list(df.paragraph_id))

    def test_separation_string_with_nl(self):
        df = Separator.separate_string('Строка\nВторая строка')
        self.assertListEqual([0, 1, 1], list(df.paragraph_id))


    def test_separator_types(self):
        df = Separator.separate_string('Слово сло'+chr(8242)+'во! Qwe - йцу '+"it's")
        self.assertListEqual(['ru','ru','punct','unk', 'punct', 'ru','unk','unk','unk'], list(df.word_type))

    def test_separator_on_emty(self):
        df = Separator.separate_string('')
        self.assertEqual(0, df.shape[0])