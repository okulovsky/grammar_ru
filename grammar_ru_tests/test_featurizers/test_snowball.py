from unittest import TestCase

from tg.grammar_ru.common import SeparatorEn
from tg.grammar_ru.features import SnowballFeaturizer

text = "John walks his old dog and goes shopping in the mall.\n It was quickly."


class SnowBallFeaturizersTestCase(TestCase):
    def test_nltk(self):
        try:
            db = SeparatorEn.build_bundle(text)
            analyzer = SnowballFeaturizer(language='eng')
            analyzer.featurize(db)
        except LookupError:
            import nltk
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')
            self.fail('Needed to download nltk data. Rerun the test')


    def test_morph_and_syntax_general(self):
        db = SeparatorEn.build_bundle(text)
        for c in ['word_id', 'sentence_id', 'paragraph_id']:
            db.src[c] += 100
        analyzer = SnowballFeaturizer(language='eng')
        analyzer.featurize(db)
        self.assertListEqual(list(range(100, 116)), list(db.snowball.index))
        self.assertListEqual(
            ['NOUN', 'VERB', 'PRON', 'ADJ', 'NOUN', 'CONJ', 'VERB', 'VERB', 'ADP', 'DET', 'NOUN', '.', 'PRON', 'VERB',
             'ADV', '.'],
            list(db.snowball.pos))

