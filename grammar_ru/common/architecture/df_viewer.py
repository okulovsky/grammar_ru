from IPython.display import HTML
from yo_fluq_ds import *

class DfViewer:
    def __init__(self, highlight_column=None, highlight_color='#ffdddd', as_html_object = True):
        self.highlight_column = highlight_column
        self.highlight_color = highlight_color
        self.as_htlm_object = as_html_object

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
        result_strs = []
        for pid in df.paragraph_id.sort_values().unique():
            result_strs.append(self._paragraph_to_html(df.loc[df.paragraph_id == pid]))
        result = '<br>'.join(result_strs)
        if self.as_htlm_object:
            return HTML(result)
        return result