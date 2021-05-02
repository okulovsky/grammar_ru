from unittest import TestCase
from grammar_ru.common import NlpAlgorithm, CombinedNlpAlgorithm
import pandas as pd

class FakeNlpAlgorithm(NlpAlgorithm):
    def __init__(self, n):
        super(FakeNlpAlgorithm, self).__init__(f'status_{n}', f'suggestion_{n}')
        self.n = n

    def run(self, df):
        df[self.get_status_column()] = df.index%self.n!=0
        df[self.get_suggest_column()] = str(self.n)

    def get_name(self):
        return str(self.n)



class NlpCombinedAlgorithmTestCase(TestCase):
    def test_combined(self):
        alg = CombinedNlpAlgorithm([
            FakeNlpAlgorithm(2),
            FakeNlpAlgorithm(3)
        ])
        df = alg.run_on_string('a b c d e f g')
        self.assertListEqual([False,True,False,False,False,True,False], list(df.status))
        self.assertListEqual(['2',None,'2','3','2',None,'2'], list(df.suggestion))
        self.assertListEqual(['2', None, '2', '3', '2', None, '2'], list(df.algorithm))


    def test_another_order(self):
        alg = CombinedNlpAlgorithm([
            FakeNlpAlgorithm(3),
            FakeNlpAlgorithm(2)
        ])
        df = alg.run_on_string('a b c d e f g')
        print(df)
        self.assertListEqual([False,True,False,False,False,True,False], list(df.status))
        self.assertListEqual(['3',None,'2','3','2',None,'3'], list(df.suggestion))
        self.assertListEqual(['3', None, '2', '3', '2', None, '3'], list(df.algorithm))

