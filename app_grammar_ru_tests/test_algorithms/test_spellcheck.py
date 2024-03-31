from unittest import TestCase
from app_grammar_ru.common.algorithms import SpellcheckAlgorithm
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None



class SpellcheckTestCase(TestCase):
    def test_spellcheck(self):
        df = SpellcheckAlgorithm().run_on_string('В этом придложении есть ашипки')
        self.assertListEqual([False,False,True,False,True], list(df.error))
        self.assertListEqual([True,True,False,True,False], list(df.suggest.isnull()))


    def test_spellcheck_multiline(self):
        df = SpellcheckAlgorithm().run_on_string('В этом придложении есть ашипки\nВ этом нет.')
        self.assertListEqual([False, False, True, False, True, False, False, False, False], list(df.error))
        self.assertListEqual([True, True, False, True, False, True, True, True, True], list(df.suggest.isnull()))

    def test_on_empty(self):
        df = SpellcheckAlgorithm().run_on_string('')
        self.assertEqual(0, df.shape[0])

    def test_with_no_request(self):
        df = SpellcheckAlgorithm().run_on_string('Здесь неот ошибок', [])
        self.assertEqual(3, df.shape[0])
        self.assertListEqual([False]*3, list(df.error))
