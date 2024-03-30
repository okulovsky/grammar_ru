from unittest import TestCase
from app_grammar_ru.common.algorithms import NlpAlgorithm
from grammar_ru.common import Separator
import pandas as pd

class FakeNlpAlgorithm(NlpAlgorithm):
    def __init__(self, n, only_errors = False):
        self.n = n
        self.only_erors = only_errors


    def _run_inner(self, db, index):
        df = pd.DataFrame({}, index=db.src.index)
        df[NlpAlgorithm.Error] = df.index % self.n == 0
        df[NlpAlgorithm.Algorithm] = str(self.n)
        df[NlpAlgorithm.Hint] = str(self.n)
        df[NlpAlgorithm.Suggest] = str(self.n)
        if self.only_erors:
            df = df.loc[df[NlpAlgorithm.Error]]
        return df



class NlpCombinedAlgorithmTestCase(TestCase):
    def test_combined(self):
        algs = [FakeNlpAlgorithm(2),FakeNlpAlgorithm(3)]
        db = Separator.build_bundle('a b c d e f g')
        df = NlpAlgorithm.combine_algorithms(db, db.src.index, *algs)
        self.assertListEqual([True, False, True, True, True, False, True], list(df.error))
        self.assertListEqual(['2', None, '2', '3', '2', None, '2'], list(df.suggest))
        self.assertListEqual(['2', None, '2', '3', '2', None, '2'], list(df.hint))
        self.assertListEqual(['2', None, '2', '3', '2', None, '2'], list(df.algorithm))

    def test_another_order(self):
        algs = [FakeNlpAlgorithm(3), FakeNlpAlgorithm(2)]
        db = Separator.build_bundle('a b c d e f g')
        df = NlpAlgorithm.combine_algorithms(db, db.src.index, *algs)
        print(df.columns)
        self.assertListEqual([True, False, True, True, True, False, True], list(df.error))
        self.assertListEqual(['3', None, '2', '3', '2', None, '3'], list(df.suggest))
        self.assertListEqual(['3', None, '2', '3', '2', None, '3'], list(df.hint))
        self.assertListEqual(['3', None, '2', '3', '2', None, '3'], list(df.algorithm))

    def test_combined_with_missing_rows(self):
        algs = [FakeNlpAlgorithm(2, True),FakeNlpAlgorithm(3, True)]
        db = Separator.build_bundle('a b c d e f g')
        df = NlpAlgorithm.combine_algorithms(db, db.src.index, *algs)
        self.assertListEqual([True, False, True, True, True, False, True], list(df.error))
        self.assertListEqual(['2', None, '2', '3', '2', None, '2'], list(df.suggest))
        self.assertListEqual(['2', None, '2', '3', '2', None, '2'], list(df.hint))
        self.assertListEqual(['2', None, '2', '3', '2', None, '2'], list(df.algorithm))

