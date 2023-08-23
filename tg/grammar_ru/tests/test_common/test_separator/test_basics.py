from unittest import TestCase
from tg.grammar_ru.common.separator import Separator, SeparatorEn, _generate_offsets
import pandas as pd


class SeparatorTestCase(TestCase):
    def test_generate_offsets(self):
        text = '  A BCD   EFG  H '
        tails, length = _generate_offsets(text, ['A', 'BCD', 'EFG'])
        self.assertListEqual([1, 3, 2], tails)
        self.assertEqual(15, length)

    def test_separation_columns(self):
        test_cases = (
            {'lang': 'ru', 'text': '«Какой-нибудь»   текст —  с знаками… И еще словами!.. Вот так.', 'sep': Separator},
            {'lang': 'en', 'text': '«cream-covered»   cake —  with signs, And words!.. Like that.', 'sep': SeparatorEn}
        )
        for case in test_cases:
            with self.subTest(msg=case['lang']):
                df = case['sep'].separate_string(case['text'])
                self.assertListEqual(Separator.COLUMNS, list(df.columns))

    def test_separation(self):
        test_cases = (
            {'lang': 'ru', 'text': '«Какой-нибудь»   текст —  с знаками… И еще словами!.. Вот так.', 'sep': Separator,
             'ans': [[0, 0, 3, 1, 2, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 12, 1, 5, 1, 1, 7, 1, 1, 3, 7, 3, 3, 3, 1]]},
            {'lang': 'en', 'text': '“cream-covered“   cake —  with signs, And words!. Like that.', 'sep': SeparatorEn,
             'ans': [[0, 0, 3, 1, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 13, 1, 4, 1, 4, 5, 1, 3, 5, 1, 1, 4, 4, 1]]}
        )
        for case in test_cases:
            with self.subTest(msg=case['lang']):
                df = case['sep'].separate_string(case['text'])
                self.assertListEqual(list(df.word_tail), case['ans'][0])
                self.assertListEqual(list(df.word_length), case['ans'][1])

    def test_separation_multi(self):
        test_cases = (
            {'lang': 'ru', 'text': ['первое предожение. Второе.', 'Второй параграф'], 'sep': Separator},
            {'lang': 'en', 'text': ['first sentence. Second.', 'Second paragraph'], 'sep': SeparatorEn}
        )
        for case in test_cases:
            with self.subTest(name=case['lang']):
                df = case['sep'].separate_paragraphs(case['text'])
                self.assertListEqual([0, 0, 0, 1, 1, 2, 2], list(df.sentence_id))
                self.assertListEqual([0, 1, 2, 3, 4, 5, 6], list(df.word_id))
                self.assertListEqual([0, 0, 0, 0, 0, 1, 1], list(df.paragraph_id))

    def test_separation_string_with_nl(self):
        test_cases = (
            {'lang': 'ru', 'text': 'Строка\nВторая строка', 'sep': Separator},
            {'lang': 'en', 'text': 'String\nSecond string', 'sep': SeparatorEn}
        )
        for case in test_cases:
            with self.subTest(name=case['lang']):
                df = case['sep'].separate_string(case['text'])
                self.assertListEqual([0, 1, 1], list(df.paragraph_id))

    def test_separator_types(self):
        test_cases = (
            {'lang': 'ru', 'text': 'Слово сло' + chr(8242) + 'во! Qwe - йцу ' + "it's", 'sep': Separator,
             'ans': ['ru', 'ru', 'punct', 'unk', 'punct', 'ru', 'unk', 'unk', 'unk']},
            {'lang': 'en', 'text': 'Word wo' + 'qe! Qwe - йцу ' + "it's", 'sep': SeparatorEn,
             'ans': ['en', 'en', 'punct', 'en', 'punct', 'unk', 'en', 'en']}
        )
        for case in test_cases:
            with self.subTest(name=case['lang']):
                df = case['sep'].separate_string(case['text'])
                self.assertListEqual(case['ans'], list(df.word_type))

    def test_separator_on_emty(self):
        test_cases = (
            {'lang': 'ru', 'sep': Separator},
            {'lang': 'en', 'sep': SeparatorEn},
        )
        for case in test_cases:
            with self.subTest(name=case['lang']):
                df = case['sep'].separate_string('')
                self.assertEqual(0, df.shape[0])
