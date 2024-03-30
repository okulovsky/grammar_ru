from tg.grammar_ru.common import Separator
from tg.grammar_ru.features import PyMorphyFeaturizer
from unittest import TestCase

text = 'Он подошел к двери. За ней никого не было.'


class CombinedNatashaAnalyzerTestCase(TestCase):
    def test_pymorphy(self):
        db = Separator.build_bundle(text)
        db.src.word_id+=100
        PyMorphyFeaturizer().featurize(db)
        self.assertListEqual([100,101,102,103,104,105,106, 107, 108, 109, 110], list(db.pymorphy.index))
        self.assertListEqual(['он', 'подойти', 'к', 'дверь', '.', 'за', 'она', 'никто', 'не', 'быть', '.'], list(db.pymorphy.normal_form))
        self.assertListEqual(
            ['normal_form', 'alternatives', 'score', 'delta_score', 'POS', 'animacy', 'gender', 'number', 'case', 'aspect', 'transitivity', 'person', 'tense', 'mood', 'voice', 'involvement'],
            list(db.pymorphy.columns))
