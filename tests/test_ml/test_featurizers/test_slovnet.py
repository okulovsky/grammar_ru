from grammar_ru.common import Separator
from grammar_ru.ml.features import SlovnetFeaturizer
from unittest import TestCase
import numpy as np

text = 'Он подошел к двери. За ней никого не было.'


class CombinedNatashaAnalyzerTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(CombinedNatashaAnalyzerTestCase, cls).setUpClass()
        cls.analyzer = SlovnetFeaturizer()
        df = Separator.separate_string(text)
        for c in ['word_id','sentence_id','paragraph_id']:
            df[c] +=100
        cls.result = cls.analyzer.featurize(df)
        print(cls.result)

    def test_morph_and_syntax_general(self):
        self.assertListEqual([100,101,102,103,104,105,106, 107, 108, 109, 110], list(self.result.index))
        self.assertListEqual([101, -1, 103, 101, 101, 106, 109, 109, 109, -1, 109], list(self.result.syntax_parent_id))
        self.assertListEqual(['nsubj', 'root', 'case', 'obl', 'punct', 'case', 'obl', 'nsubj', 'advmod', 'root', 'punct'], list(self.result.relation))
        self.assertListEqual(['PRON', 'VERB', 'ADP', 'NOUN', 'PUNCT', 'ADP', 'PRON', 'PRON', 'PART', 'AUX', 'PUNCT'], list(self.result.POS))
        self.assertListEqual(
            ['POS', 'Case', 'Gender', 'Number', 'Person', 'Aspect', 'Mood', 'Tense', 'VerbForm', 'Voice', 'Animacy','Polarity', 'relation', 'syntax_parent_id'],
            list(self.result.columns)
        )
