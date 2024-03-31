import typing as tp
from pathlib import Path
from unittest import TestCase

import pandas as pd
import numpy as np

from grammar_ru.common import Separator, DataBundle
from app_grammar_ru.ml.alternative import WordSequenceFilterer, WordPairsNegativeSampler

def ChtobyFilterer():
    return WordSequenceFilterer([['что', "бы"], ["чтобы"]])

def ChtobyNegativeSampler():
    return WordPairsNegativeSampler([('чтобы', "что бы"), ('Чтобы', 'Что бы')])

class ChtobyTestCase(TestCase):
    def test_filterer(self):
        db = Separator.build_bundle(
            '''Чтобы приготовить суп. Нужно купить продукты. 
            Что бы мне сделать. С Новым Годом!. Для того, чтобы. 
            Во что бы мне поиграть.'''
        )
        frame = db.data_frames['src']
        expected_words = [
            'Чтобы', 'приготовить', 'суп',
            '.', 'Что', 'бы',
            'мне', 'сделать', '.', 
            'Для', 'того', ',', 
            'чтобы', '.', 
            'Во', 'что', 'бы', 
            'мне', 'поиграть', '.'
        ]

        filterer = ChtobyFilterer()
        filterered_df = filterer.get_filtered_df(frame)

        self.assertListEqual(expected_words, list(filterered_df['word']))

    def test_negative_sampler(self):
        self.maxDiff = None
        db = Separator.build_bundle(
            '''Чтобы приготовить суп. 
            Что бы мне сделать. Для того, чтобы. 
            Во что бы мне поиграть, чтобы развлечься.'''
        )
        frame = db.data_frames['src']
        filterer = ChtobyFilterer()
        filterered_df = filterer.get_filtered_df(frame)

        sampler = ChtobyNegativeSampler()
        negative = pd.concat(sampler.build_all_negative_samples_from_positive(filterered_df))
        expected = '''Что бы приготовить суп . Чтобы мне сделать . Для того , что бы . Во чтобы мне поиграть , чтобы развлечься . Во что бы мне поиграть , что бы развлечься .'''
        expected_array = expected.split(' ')
        self.assertListEqual(expected_array, list(negative.word))

