import typing as tp
from pathlib import Path
from unittest import TestCase

import pandas as pd
import numpy as np

from tg.grammar_ru.common import Separator, DataBundle
from tg.grammar_ru.ml.tasks.train_index_builder.sentence_filterer import ChtobyFilterer
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import ChtobyIndexBuilder
from tg.grammar_ru.ml.tasks.train_index_builder.negative_sampler import ChtobyNegativeSampler


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
        db = Separator.build_bundle(
            '''Чтобы приготовить суп. 
            Что бы мне сделать. Для того, чтобы. 
            Во что бы мне поиграть, чтобы развлечься.'''
        )
        frame = db.data_frames['src']
        expected_words = [
            'Что', 'бы', 'приготовить',
            'суп', '.', 'Чтобы',
            'мне', 'сделать', '.',
            'Для', 'того', ',',
            'что', 'бы', '.',
            'Во', 'чтобы', 'мне',
            'поиграть', ',', 'что',
            'бы', 'развлечься', '.',
            'Во', 'что', 'бы',
            'мне', 'поиграть', ',',
            'что', 'бы', 'развлечься',
            '.', 'Во', 'чтобы',
            'мне', 'поиграть', ',',
            'чтобы', 'развлечься', '.'
        ]

        sampler = ChtobyNegativeSampler()
        negative = sampler.build_negative_sample_from_positive(frame)

        self.assertListEqual(expected_words, list(negative['word']))
        # TODO: check targets

    def test_algorithm(self):
        pass
