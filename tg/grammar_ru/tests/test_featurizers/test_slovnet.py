from tg.grammar_ru.common import Separator
from tg.grammar_ru.ml.features import SlovnetFeaturizer
from unittest import TestCase
import numpy as np
from tg.common import DataBundle

text = 'Он подошел к двери. За ней никого не было.'


class SlovnetFeaturizersTestCase(TestCase):
    def test_morph_and_syntax_general(self):
        db = Separator.build_bundle(text)
        for c in ['word_id', 'sentence_id', 'paragraph_id']:
            db.src[c] += 100
        analyzer = SlovnetFeaturizer()
        analyzer.featurize(db)

        self.assertListEqual([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], list(db.slovnet.index))
        self.assertListEqual([101, -1, 103, 101, 101, 106, 109, 109, 109, -1, 109], list(db.slovnet.syntax_parent_id))
        self.assertListEqual(['nsubj', 'root', 'case', 'obl', 'punct', 'case', 'obl',
                              'nsubj', 'advmod', 'root', 'punct'], list(db.slovnet.relation))
        self.assertListEqual(['PRON', 'VERB', 'ADP', 'NOUN', 'PUNCT', 'ADP', 'PRON',
                              'PRON', 'PART', 'AUX', 'PUNCT'], list(db.slovnet.POS))
        self.assertListEqual(
            ['POS', 'Case', 'Gender', 'Number', 'Person', 'Aspect', 'Mood', 'Tense',
                'VerbForm', 'Voice', 'Animacy', 'Polarity', 'relation', 'syntax_parent_id'],
            list(db.slovnet.columns)
        )
