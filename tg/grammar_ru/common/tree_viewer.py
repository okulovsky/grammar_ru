from typing import *
import pandas as pd
import pygraphviz as pgv
from yo_fluq_ds import *
from IPython.display import Image

class TreeViewer:
    def __init__(self, sentence_id: int, src_table: pd.DataFrame):
        self.src_table = src_table.loc[src_table.sentence_id==sentence_id].set_index('word_id')[['sentence_id','word']]

    def _self_merge(self, series, name, default):
        column = self.src_table[[]].merge(series.to_frame(name), left_index=True, right_index=True, how='left')
        column = column.fillna(default)
        self.src_table[name] = column

    def add_relation(self, syntax_parent_id: pd.Series, relation_name: Optional[pd.Series]=None) -> 'TreeViewer':
        self._self_merge(syntax_parent_id,'parent',-2)
        if relation_name is not None:
            self._self_merge(relation_name, 'relation', '')
        else:
            self.src_table.relation = ''
        return self

    def add_edge_color(self, color:pd.Series, mapping=None):
        self._self_merge(color,'edge_color','black')
        if mapping is not None:
            self.src_table.edge_color = self.src_table.edge_color.replace(mapping)
        return self

    def add_labels(self, labels: pd.Series):
        self._self_merge(labels, 'label', 'XXX')
        return self


    def draw(self):
        G = pgv.AGraph(strict=False, directed=True)
        for row in Query.df(self.src_table.reset_index()):
            G.add_node(row.word_id, label=row.get('label',row.word))

        if 'parent' in self.src_table.columns:
            for row in Query.df(self.src_table.reset_index()):
                G.add_edge(
                    row.word_id,
                    row.parent,
                    label=row.get('relation',''),
                    color=row.get('edge_color', 'black')
                )
        G.layout(prog="dot")
        G.draw("file.png")
        return Image('file.png')
