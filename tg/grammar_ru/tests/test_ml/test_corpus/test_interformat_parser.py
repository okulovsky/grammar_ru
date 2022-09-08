from tg.grammar_ru.ml.corpus.formats import InterFormatParser
from unittest import TestCase


class MdParserTestCase(TestCase):
    def test_circumvention(self):
        f = InterFormatParser._circumvent_separator_problems
        self.assertEqual('Тест тест',f('Тест тест'))
        self.assertEqual('Тест', f('Те'+chr(173)+'ст'))
        self.assertEqual("It's not й′цц′у", f("It's not й'цц'у"))