from grammar_ru.common import Separator
from grammar_ru.features import MorphemeTikhonovFeaturizer, PyMorphyFeaturizer
from unittest import TestCase

class TikhonovTextCase(TestCase):
    def test_tikhonov(self):
        db = Separator.build_bundle("Безответственное пароходство")
        PyMorphyFeaturizer().featurize(db)
        MorphemeTikhonovFeaturizer().featurize(db)
        self.assertListEqual(['без', 'ответственн', 'ый', 'пар', 'о', 'ход', 'ств', 'о'], list(db.tikhonov_morphemes.morpheme))
        self.assertListEqual(['PREF', 'ROOT', 'END', 'ROOT', 'LINK', 'ROOT', 'SUFF', 'END'], list(db.tikhonov_morphemes.morpheme_type))


