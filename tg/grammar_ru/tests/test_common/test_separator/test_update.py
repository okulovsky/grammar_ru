from unittest import TestCase
from yo_fluq_ds import *
from tg.grammar_ru.common import Separator

text = '''
Нулевое. Оригинальное
Первое. Оригинальное
Второе. Оригинальное
Третье. Оригинальное
Четвертое. Оригинальное
Пятое. Оригинальное
Шестое. Оригинальное
'''

class SeparateUpdateTextCase(TestCase):
    def test_update(self):
        pars = Query.en(text.split('\n')).where(lambda z: z != '').to_list()
        df = Separator.separate_paragraphs(pars)

        new_pars = [
            pars[0],
            pars[1],
            'Нулевая. Вставка',
            pars[4],
            pars[3],
            'Первая. Вставка',
            'Вторая. Вставка',
            pars[6],
            'Третья. Вставка'
        ]
        df1 = Separator.separate_paragraphs(new_pars)

        df2 = Separator.update_separation(df, new_pars, [0, 1, None, 4, 3, None, None, 6, None])

        for c in df1.columns:
            self.assertListEqual(list(df1[c]), list(df2[c]))

        pd.options.display.width = None

        self.assertListEqual(
            [False, False, False, False, False, False, True, True, True, False, False, False, False, False, False, True,
             True, True, True, True, True, False, False, False, True, True, True],
            list(df2.updated)
        )
        self.assertListEqual(
            [0, 1, 2, 3, 4, 5, -1, -1, -1, 12, 13, 14, 9, 10, 11, -1, -1, -1, -1, -1, -1, 18, 19, 20, -1, -1, -1],
            list(df2.original_word_id)
        )
        self.assertListEqual(
            [0, 0, 1, 2, 2, 3, -1, -1, -1, 8, 8, 9, 6, 6, 7, -1, -1, -1, -1, -1, -1, 12, 12, 13, -1, -1, -1],
            list(df2.original_sentence_id)
        )
        self.assertListEqual(
            [0, 0, 0, 1, 1, 1, -1, -1, -1, 4, 4, 4, 3, 3, 3, -1, -1, -1, -1, -1, -1, 6, 6, 6, -1, -1, -1],
            list(df2.original_paragraph_id)
        )

