from grammar_ru.common import Separator
from grammar_ru.ml.features import PyMorphyFeaturizer
from unittest import TestCase
import numpy as np

text = 'Он подошел к двери. За ней никого не было.'


class CombinedNatashaAnalyzerTestCase(TestCase):
    def test_pymorphy(self):
        df = Separator.separate_string(text)
        df.word_id += 100
        result = PyMorphyFeaturizer().featurize(df)
        self.assertListEqual([100,101,102,103,104,105,106, 107, 108, 109, 110], list(result.index))
        self.assertListEqual(['он', 'подойти', 'к', 'дверь', '.', 'за', 'она', 'никто', 'не', 'быть', '.'], list(result.normal_form))
        self.assertListEqual(
            ['normal_form', 'alternatives', 'score', 'delta_score', 'POS', 'animacy', 'gender', 'number', 'case', 'aspect', 'transitivity', 'person', 'tense', 'mood', 'voice', 'involvement'],
            list(result.columns))
