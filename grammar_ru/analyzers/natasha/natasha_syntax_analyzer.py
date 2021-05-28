from .natasha_analyzer import NatashaAnalyzer
from .natasha_navec import NatashaNavec
import pandas as pd
from slovnet import Syntax
from typing import *
import os


class NatashaSyntaxAnalyzer(NatashaAnalyzer):
    def __init__(self):
        super(NatashaAnalyzer, self).__init__()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/slovnet_syntax_news_v1.tar')
        self.syntax = Syntax.load(path)
        self.syntax.navec(NatashaNavec.get_navec())

    def analyze_chunks(self, df: pd.DataFrame, chunks: List[List[str]]) -> pd.DataFrame:
        syntax_chunks = []
        counter = 0
        for i, syntax_res in enumerate(self.syntax.map(chunks)):
            for j, syntax_token in enumerate(syntax_res.tokens):
                syntax_chunks.append({})

                relative_parent_id = int(syntax_token.head_id) - 1
                if relative_parent_id >= 0:
                    absolute_parent_id = df.loc[
                        (df['sentence_id'] == i) & (df['word_index'] == relative_parent_id)
                    ]['word_id'].item()

                    syntax_chunks[-1]["parent_id"] = absolute_parent_id
                    syntax_chunks[-1]["rel"] = syntax_token.rel
                else:
                    syntax_chunks[-1]["parent_id"] = -1

                syntax_chunks[-1]['word_id'] = counter
                counter += 1

        return pd.DataFrame(syntax_chunks)
