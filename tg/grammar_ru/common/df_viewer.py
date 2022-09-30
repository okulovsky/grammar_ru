from typing import *
from IPython.display import HTML
from yo_fluq_ds import *

class Fragment:
    def __init__(self, word: str, tail: int):
        self.word = word
        self.tail = tail
        self.color = None


class Highlight:
    def __init__(self, column_name: str, value_to_color: Dict[Any, str]):
        self.column_name = column_name
        self.value_to_color = value_to_color

    def style_row(self, row: Dict, fragment: Fragment):
        if self.column_name not in row:
            raise ValueError(f"Column {self.column_name} is not in df's columns")
        val = row[self.column_name]
        for key, color in self.value_to_color.items():
            if val == key:
                fragment.color = color


class DfViewer:
    def __init__(self):
        self.highlights = []

    def highlight(self, column_name: str, value_to_color: Dict[Any, str]) -> 'ReverseSeparator':
        self.highlights.append(Highlight(column_name, value_to_color))
        return self

    def _paragraph_to_html(self, df):
        df = df.sort_values('word_id')
        fragments = []
        for row in Query.df(df):
            fragment = Fragment(row.word, row.word_tail)
            for h in self.highlights:
                h.style_row(row, fragment)
            fragments.append(fragment)
        result = []
        for f in fragments:
            if f.color is not None:
                result.append(f'<span style="background-color:{f.color};">')
            result.append(f.word)
            if f.color is not None:
                result.append('</span>')
            result.append(' ' * f.tail)
        return ''.join(result)

    def to_html(self, df):
        pars = df.drop_duplicates('paragraph_id').sort_values('paragraph_id')
        pars = ["<p>" + self._paragraph_to_html(df.loc[df.paragraph_id == p]) + "</p>" for p in pars.paragraph_id]
        html = ''.join(pars)
        return html

    def to_html_display(self, df):
        html = self.to_html(df)
        return HTML(html)

    def _paragraph_to_text(self, df):
        df = df.sort_values('word_id')
        tail = pd.Series(' ', index=df.index) * df.word_tail
        return ''.join(df.word + tail)

    def to_text(self, df):
        pars = df.drop_duplicates('paragraph_id').sort_values('paragraph_id')
        pars = [self._paragraph_to_text(df.loc[df.paragraph_id == p]) for p in pars.paragraph_id]
        return '\n'.join(pars)
