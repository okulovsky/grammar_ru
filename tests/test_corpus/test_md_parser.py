from grammar_ru.corpus.formats.md_parser import MdParser
from unittest import TestCase
import re

from grammar_ru.common.architecture.separator import Symbols
class MdParserTestCase(TestCase):
    def test_circumvention(self):
        f = MdParser._circumvent_separator_problems
        self.assertEqual('Тест тест',f('Тест тест'))
        self.assertEqual('Тест', f('Те'+chr(173)+'ст'))
        self.assertEqual("It's not й′цц′у", f("It's not й'цц'у"))