from grammar_ru import RepetitionsAlgorithm
from unittest import TestCase
import pandas as pd

text = 'Она была она красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'
pd.options.display.max_columns = None
pd.options.display.width = None

class RepetitionsAlgorithmTestCase(TestCase):
    def test_simple(self):
        df = RepetitionsAlgorithm(50, True, False, False).run_on_string(text)
        self.assertListEqual([2,10], list(df.loc[~df.repetition_status].word_id))


    def test_normal(self):
        df = RepetitionsAlgorithm(50, False, True, False).run_on_string(text)
        print(df)
        self.assertListEqual([2,7, 10], list(df.loc[~df.repetition_status].word_id))

    def test_tikhonov(self):
        df = RepetitionsAlgorithm(50, False, False, True).run_on_string(text)
        self.assertListEqual([2,5, 7, 10, 12, 14], list(df.loc[~df.repetition_status].word_id))
