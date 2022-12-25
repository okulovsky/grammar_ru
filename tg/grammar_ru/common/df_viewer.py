from typing import *
from IPython.display import HTML
from yo_fluq_ds import *
from enum import Enum
import matplotlib
import numpy as np

class Fragment:
    def __init__(self, word: str, tail: int):
        self.word = word
        self.tail = tail
        self.background_color = None
        self.foreground_color = None
        self.tooltip = None

    def generate_opening_span(self):
        if self.background_color is None and self.foreground_color is None and self.tooltip is None:
            return ''
        result = '<span'

        if self.background_color is not None or self.foreground_color is not None:
            result+=' style="'
            if self.background_color is not None:
                result+=f'background-color:{self.background_color};'
            if self.foreground_color is not None:
                result+=f'color:{self.foreground_color};'
            result+='"'

        if self.tooltip is not None:
            result+=f' title="{self.tooltip}"'

        result+='">'
        return result


    def generate_closing_span(self):
        if self.background_color is None and self.foreground_color is None:
            return ''
        return '</span>'

class HighlightType(Enum):
    Foreground = 0
    Background = 1
    Tooltip = 2


class Highlight:
    def __init__(self,
                 type: HighlightType,
                 column_name: str,
                 value_to_color: Optional[Dict[Any, str]] = None,
                 gradient_between: Optional[Tuple[str,str]] = None
                 ):
        self.type = type
        self.column_name = column_name
        self.value_to_color = value_to_color
        self.gradient_between = gradient_between
        if self.gradient_between is not None:
            self.gradient_from = np.array(matplotlib.colors.to_rgb(self.gradient_between[0]))
            self.gradient_to = np.array(matplotlib.colors.to_rgb(self.gradient_between[1]))


    def preview(self, df):
        if self.gradient_between is not None:
            self.min_value = df[self.column_name].min()
            self.max_value = df[self.column_name].max()


    def _get_color(self, val):
        if self.value_to_color is not None:
            if val in self.value_to_color:
                return self.value_to_color[val]
            return None
        if self.gradient_between is not None:
            v = (val-self.min_value)/(self.max_value-self.min_value)
            c = self.gradient_from*(1-v) + self.gradient_to*v
            return matplotlib.colors.to_hex(c)


    def style_row(self, row: Dict, fragment: Fragment):
        if self.column_name not in row:
            raise ValueError(f"Column {self.column_name} is not in df's columns")
        val = row[self.column_name]
        if self.type == HighlightType.Foreground:
            fragment.foreground_color = self._get_color(val)
        elif self.type == HighlightType.Background:
            fragment.background_color = self._get_color(val)
        elif self.type == HighlightType.Tooltip:
            fragment.tooltip = str(val)


class DfViewer:
    def __init__(self):
        self.highlights = []

    def highlight(self,
                  column_name: str,
                  value_to_color: Optional[Dict[Any, str]] = None,
                  gradient_between: Optional[Tuple[str, str]] = None) -> 'DfViewer':
        self.highlights.append(Highlight(HighlightType.Background, column_name, value_to_color, gradient_between))
        return self

    def color(self,
              column_name: str,
              value_to_color: Optional[Dict[Any, str]] = None,
              gradient_between: Optional[Tuple[str, str]] = None) -> 'DfViewer':
        self.highlights.append(Highlight(HighlightType.Foreground, column_name, value_to_color, gradient_between))
        return self

    def tooltip(self, column_name) -> 'DfViewer':
        self.highlights.append(Highlight(HighlightType.Tooltip, column_name))
        return self

    def _paragraph_to_html(self, df):
        df = df.sort_values('word_id')
        for h in self.highlights:
            h.preview(df)
        fragments = []
        for row in Query.df(df):
            fragment = Fragment(row.word, row.word_tail)
            for h in self.highlights:
                h.style_row(row, fragment)
            fragments.append(fragment)
        result = []
        for f in fragments:
            result.append(f.generate_opening_span())
            result.append(f.word)
            result.append(f.generate_closing_span())
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

    def to_sentences_strings(self, df, group_by_column='sentence_id'):
        df = df.assign(space=' ')
        return df.assign(word_print = df.word+df.space*df.word_tail).groupby(group_by_column).word_print.sum()