from unittest import TestCase
from grammar_ru.features import DialogMarkupFeaturizer
from grammar_ru.common import Separator, DataBundle


def _build_summary(df, markup):
    vdf = df.merge(markup, left_on='word_id', right_index=True).copy()
    vdf = vdf[['word', 'dialog_type']]
    vdf['next_dialog_type'] = vdf.dialog_type.shift(-1).fillna('')
    vdf['next_word'] = vdf.word.shift(-1).fillna('')
    vdf = vdf.loc[vdf.dialog_type != vdf.next_dialog_type]
    vdf = vdf[['word', 'next_word', 'dialog_type', 'next_dialog_type']]
    return [list(z[1]) for z in vdf.iterrows()]


class DialogFeaturizerIntegrationTestCase(TestCase):
    def check(self, s, reference):
        df = Separator.separate_string(s)
        db = DataBundle(src=df)
        featurizer = DialogMarkupFeaturizer()
        featurizer.featurize(db)
        summary = _build_summary(db.src, db.dialog_markup)
        self.assertListEqual(reference, summary)


    def test_no_dialog(self):
        self.check(
            'Он вышел.',
            [['.', '', 'no', '']]
        )

    def test_no_dialog_with_borderline_dash(self):
        self.check(
            'Те, которые отказались, — нет.',
            [['.', '', 'no', '']]
        )

    def test_dialog_1(self):
        self.check(
            '— О, всему понемножку, — беззаботно ответил Дамблдор.',
            [['—', 'О', 'dialog-dash', 'speech'],
             ['понемножку', ',', 'speech', 'dialog-symbol'],
             [',', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'беззаботно', 'dialog-dash', 'action'],
             ['.', '', 'action', '']]
        )

    def test_dialog_2(self):
        self.check(
            '— В целом, это мудрое решение, — сказал Дамблдор. — Впрочем, я думаю, ты мог бы сделать исключение для своих друзей, мистера Рональда Уизли и мисс Гермионы Грэйнджер. Да, — подтвердил он, видя изумление Гарри, — я считаю, что им следует об этом знать. Ты оказываешь им плохую услугу, скрывая от них такую важную вещь.',
            [['—', 'В', 'dialog-dash', 'speech'],
             ['решение', ',', 'speech', 'dialog-symbol'],
             [',', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'сказал', 'dialog-dash', 'action'],
             ['Дамблдор', '.', 'action', 'dialog-symbol'],
             ['.', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'Впрочем', 'dialog-dash', 'speech'],
             ['Да', ',', 'speech', 'dialog-symbol'],
             [',', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'подтвердил', 'dialog-dash', 'action'],
             ['Гарри', ',', 'action', 'dialog-symbol'],
             [',', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'я', 'dialog-dash', 'speech'],
             ['.', '', 'speech', '']]
        )

    def test_dialog_3(self):
        self.check(
            '— Нет! — надтреснутым голосом вскрикнул торговец. Он постоянно вертел головой, будто стараясь увидеть все происходящее на улице за спиной Ранда. — Не упоминай... — Фейн понизил голос до хриплого шепота и повернул голову, бросая на Ранда быстрые, косые взгляды. — ...их. В городе Белоплащники.',
            [['—', 'Нет', 'dialog-dash', 'speech'],
             ['Нет', '!', 'speech', 'dialog-symbol'],
             ['!', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'надтреснутым', 'dialog-dash', 'action'],
             ['Ранда', '.', 'action', 'dialog-symbol'],
             ['.', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'Не', 'dialog-dash', 'speech'],
             ['упоминай', '...', 'speech', 'dialog-symbol'],
             ['...', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'Фейн', 'dialog-dash', 'action'],
             ['взгляды', '.', 'action', 'dialog-symbol'],
             ['.', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', '...', 'dialog-dash', 'speech'],
             ['.', '', 'speech', '']]
        )

    def test_special_symbol_1(self):
        self.check(
            '— При чем тут «ненавидит»? — сердитым шепотом возразила миссис Уизли. — Просто я считаю, что они поторопились с помолвкой, вот и все!',
            [['—', 'При', 'dialog-dash', 'speech'],
             ['»', '?', 'speech', 'dialog-symbol'],
             ['?', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'сердитым', 'dialog-dash', 'action'],
             ['Уизли', '.', 'action', 'dialog-symbol'],
             ['.', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'Просто', 'dialog-dash', 'speech'],
             ['!', '', 'speech', '']]
        )

    def test_special_symbol_2(self):
        self.check(
            '— А насчет Перси что слышно? — спросил Гарри (третий по старшинству из братьев Уизли рассорился с семьей). — Он так и не общается с твоими родителями?',
            [['—', 'А', 'dialog-dash', 'speech'],
             ['слышно', '?', 'speech', 'dialog-symbol'],
             ['?', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'спросил', 'dialog-dash', 'action'],
             [')', '.', 'action', 'dialog-symbol'],
             ['.', '—', 'dialog-symbol', 'dialog-dash'],
             ['—', 'Он', 'dialog-dash', 'speech'],
             ['?', '', 'speech', '']]
        )






