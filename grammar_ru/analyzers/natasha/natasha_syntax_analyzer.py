from .natasha_analyzer import NatashaAnalyzer
import pandas as pd
from slovnet import Syntax
from navec import Navec
from typing import *


class NatashaSyntaxAnalyzer(NatashaAnalyzer):
    def __init__(self):
        # TODO: Specify paths
        # TODO: Add required columns
        self.navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
        self.syntax = Syntax.load('slovnet_syntax_news_v1.tar')
        self.syntax.navec(self.navec)
        super(NatashaAnalyzer, self).__init__()

    def analyze_chunks(self, df: pd.DataFrame, chunks: List[List[str]]) -> pd.DataFrame:
        syntax_chunks = []
        for i, syntax_res in enumerate(self.syntax.map(chunks)):
            syntax_chunks.append({})
            for j, syntax_token in enumerate(syntax_res.tokens):
                relative_parent_id = int(syntax_token.head_id) - 1
                absolute_parent_id = df.loc[
                    (df['sentence_id'] == i) & (df['word_index'] == relative_parent_id)
                ].at['word_id']
                syntax_chunks[-1]["parent_id"] = absolute_parent_id
                syntax_chunks[-1]["rel"] = syntax_token.rel

        # TODO: List of dictionaries to rows
        raise NotImplementedError()
