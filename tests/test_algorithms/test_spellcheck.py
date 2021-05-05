from unittest import TestCase
from grammar_ru.algorithms.spellcheck import SpellcheckAlgorithm


class SpellcheckTestCase(TestCase):
    def test_spellcheck(self):
        df = SpellcheckAlgorithm().run_on_string('В этом придложении есть ашипки')
        self.assertListEqual([True, True, False, True, False], list(df.spellcheck_status))
        self.assertListEqual([True, True, False, True, False], list(df.spellcheck_suggestion.isnull()))
