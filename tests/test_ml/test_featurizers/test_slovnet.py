from grammar_ru.ml.features.slovnet_context_featurizer import SlovnetContextFeaturizer
from grammar_ru.common import Separator
from grammar_ru.ml.features import SlovnetFeaturizer
from unittest import TestCase
import numpy as np

text = 'Он подошел к двери. За ней никого не было.'


class SlovnetFeaturizersTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(SlovnetFeaturizersTestCase, cls).setUpClass()
        cls.analyzer = SlovnetFeaturizer()
        cls.context_featurizer = SlovnetContextFeaturizer()
        df = Separator.separate_string(text)
        for c in ['word_id', 'sentence_id', 'paragraph_id']:
            df[c] += 100
        cls.result = cls.analyzer.featurize(df)
        cls.context_result = cls.context_featurizer.featurize(cls.result)
        # print(cls.result)
        # print(cls.context_result)

    def test_morph_and_syntax_general(self):
        self.assertListEqual([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], list(self.result.index))
        self.assertListEqual([101, -1, 103, 101, 101, 106, 109, 109, 109, -1, 109], list(self.result.syntax_parent_id))
        self.assertListEqual(['nsubj', 'root', 'case', 'obl', 'punct', 'case', 'obl',
                              'nsubj', 'advmod', 'root', 'punct'], list(self.result.relation))
        self.assertListEqual(['PRON', 'VERB', 'ADP', 'NOUN', 'PUNCT', 'ADP', 'PRON',
                              'PRON', 'PART', 'AUX', 'PUNCT'], list(self.result.POS))
        self.assertListEqual(
            ['POS', 'Case', 'Gender', 'Number', 'Person', 'Aspect', 'Mood', 'Tense',
                'VerbForm', 'Voice', 'Animacy', 'Polarity', 'relation', 'syntax_parent_id'],
            list(self.result.columns)
        )

    def test_context_featurizer_general(self):
        self.assertListEqual([1, 0, 0, 0, 0, 0, -1, -1, -1, -2, 1, 2, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0,
                              0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -2, 1, 0, 0, 0, 0], list(self.context_result["shift"]))
        self.assertListEqual([101, 100, 103, 104, 101, 109, 100, 103, 104, 102, 103, 101, 102, 101, 100, 103, 104, 102, 101, 100, 103, 104, 106, 109, 105, 109, 106, 107, 108,
                              110, 105, 109, 106, 107, 108, 110, 109, 106, 107, 108, 110, 101, 109, 106, 107, 108, 110, 105, 109, 106, 107, 108, 110], list(self.context_result.relative_word_id))
