from unittest import TestCase
from grammar_ru.algorithms import SpellcheckAlgorithm
import pandas as pd

pd.options.display.max_columns = None
pd.options.display.width = None



class SpellcheckTestCase(TestCase):
    def test_spellcheck(self):
        df = SpellcheckAlgorithm().run_on_string('В этом придложении есть ашипки')
        self.assertListEqual([True,True,False,True,False], list(df.spellcheck_status))
        self.assertListEqual([True,True,False,True,False], list(df.spellcheck_suggestion.isnull()))


    def test_spellcheck_multiline(self):
        df = SpellcheckAlgorithm().run_on_string('В этом придложении есть ашипки\nВ этом нет.')
        self.assertListEqual([True, True, False, True, False, True, True, True, True], list(df.spellcheck_status))
        self.assertListEqual([True, True, False, True, False, True, True, True, True], list(df.spellcheck_suggestion.isnull()))

    def test_on_empty(self):
        df = SpellcheckAlgorithm().run_on_string('')

    def test_with_no_request(self):
        df = SpellcheckAlgorithm().run_on_string('Здесь неот ошибок', [])
