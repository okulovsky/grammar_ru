from IPython.display import HTML
from yo_fluq_ds import *

class DfViewer:
    def __init__(self, highlight_column=None, highlight_color='#ffdddd'):
        self.highlight_column = highlight_column
        self.highlight_color = highlight_color

    def _paragraph_to_html(self, df):
        df = df.sort_values('word_id')
        result = ''
        offset = 0
        for row in Query.df(df):
            while offset < row.word_offset:
                result += ' '
                offset += 1

            add_highlight = self.highlight_column is not None and row[self.highlight_column]
            if add_highlight:
                result += f'<span style="background-color:{self.highlight_color};">'

            result += row.word
            offset += len(row.word)

            if add_highlight:
                result += '</span>'
        return result

    def convert(self, df):
        result = ''
        for pid in df.paragraph_id.sort_values().unique():
            result += self._paragraph_to_html(df.loc[df.paragraph_id == pid]) + ' '
        return HTML(result)