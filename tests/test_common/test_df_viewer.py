from grammar_ru import DfViewer, Separator
from unittest import TestCase

class DfViewerTestCase(TestCase):
    def test_viewer(self):
        text = 'Мама мыла раму'
        df = Separator.separate_string(text)
        v = DfViewer(as_html_object=False)
        self.assertEqual(text, v.convert(df))

    def test_highlight(self):
        text = 'Мама мыла раму'
        df = Separator.separate_string(text)
        df['highlight'] = df.word_id==1
        v = DfViewer(as_html_object=False, highlight_column='highlight')
        self.assertEqual('Мама <span style="background-color:#ffdddd;">мыла</span> раму', v.convert(df))
