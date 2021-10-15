from grammar_ru.algorithms import RepetitionsAlgorithm
from grammar_ru.common import Separator, DataBundle
from unittest import TestCase
import pandas as pd

text = 'Она была она красива. Он любил красивые вещи. Вещи, нитрокраситель и нитроэмаль!'
pd.options.display.max_columns = None
pd.options.display.width = None


class RepetitionsAlgorithmTestCase(TestCase):
    def test_simple(self):
        df = RepetitionsAlgorithm(50, True, False, False).run_on_string(text)
        self.assertListEqual([2, 10], list(df.loc[~df.repetition_status].word_id))

    def test_normal(self):
        df = RepetitionsAlgorithm(50, False, True, False).run_on_string(text)
        self.assertListEqual([2,7, 10], list(df.loc[~df.repetition_status].word_id))

    def test_tikhonov(self):
        df = RepetitionsAlgorithm(50, False, False, True).run_on_string(text)
        self.assertListEqual([2, 5, 7, 10, 12, 14], list(df.loc[~df.repetition_status].word_id))

    def test_all(self):
        df = RepetitionsAlgorithm(50, True, True, True).run_on_string(text)
        self.assertListEqual([2, 5, 7, 10, 12, 14], list(df.loc[~df.repetition_status].word_id))

    def test_multiline(self):
        df = RepetitionsAlgorithm(50, True, True, True).run_on_string("Повтор\nЕще повтор")
        self.assertListEqual([True, True, False], list(df.repetition_status))

    def test_on_empty(self):
        df = RepetitionsAlgorithm(50, True, True, True).run_on_string("Здесь нет ошибок", [])
        self.assertListEqual([True,True,True], list(df.repetition_status))

    def test_extended_mode(self):
        df = RepetitionsAlgorithm(50,True,True,True).run_on_string('Един един единый двуединый')
        self.assertListEqual([-1,0,1,2], list(df.repetition_reference))
        self.assertListEqual([None,'simple','normal','tikhonov'], list(df.repetition_algorithm))

    def test_tikhonov_join(self):
        df = RepetitionsAlgorithm(50,True,True,True).run_on_string('двуединое единообразие')
        self.assertListEqual([True,False], list(df.repetition_status))
        
        
    def test_usage_of_provided_pymorphy_column(self):
        df = Separator.separate_string("окно открыто")
        df['check_requested'] = True

        df1 = df.copy()
        alg = RepetitionsAlgorithm(50, False, True, False)
        alg.run_on_bundle(DataBundle(src=df1))
        self.assertTrue(df1.repetition_status.all())

        df2 = df.copy()
        pym = df2[['word_id']].copy()
        pym['normal_form'] = 'окно'
        pym = pym.set_index('word_id')
        alg = RepetitionsAlgorithm(50, False, True, False)
        alg.run_on_bundle(DataBundle(src=df2, pymorphy=pym))
        self.assertFalse(df2.repetition_status.all())

        df3 = df.copy()
        pym = df3[['word_id']].copy()
        pym['normal_form'] = ['двуединый','единообразие']
        pym = pym.set_index('word_id')
        alg = RepetitionsAlgorithm(50, False, False, True)
        alg.run_on_bundle(DataBundle(src=df3, pymorphy=pym))
        self.assertFalse(df3.repetition_status.all())

