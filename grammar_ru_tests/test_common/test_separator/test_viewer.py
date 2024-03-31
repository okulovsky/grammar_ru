from grammar_ru.common import Separator
from unittest import TestCase

df = Separator.separate_string('Казнить нельзя, помиловать.\nМама мыла раму.')

class DfViewerTestCase(TestCase):
    def test_text(self):
        self.assertEqual('Казнить нельзя, помиловать.\nМама мыла раму.', Separator.Viewer().to_text(df))

    def test_html(self):
        html = Separator.Viewer().highlight('word',{'.':'#ff0000'}).to_html(df)
        self.assertEqual('<p>Казнить нельзя, помиловать<span style="background-color:#ff0000;">.</span></p><p>Мама мыла раму<span style="background-color:#ff0000;">.</span></p>', html)
