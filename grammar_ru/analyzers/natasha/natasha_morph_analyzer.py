from .natasha_navec import NatashaNavec
from .natasha_analyzer import NatashaAnalyzer
import pandas as pd
from slovnet import Morph
from typing import *
import os


class NatashaMorphAnalyzer(NatashaAnalyzer):
    def __init__(self):
        super(NatashaAnalyzer, self).__init__()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/slovnet_morph_news_v1.tar')
        self.morph = Morph.load(path, batch_size=4)
        self.morph.navec(NatashaNavec.get_navec())

    def analyze_chunks(self, df: pd.DataFrame, chunks: List[List[str]]) -> pd.DataFrame:
        morph_chunks = []
        counter = 0
        for i, morph_res in enumerate(self.morph.map(chunks)):
            for j, morph_token in enumerate(morph_res.tokens):
                morph_chunks.append({})
                morph_chunks[-1]['word_id'] = counter
                morph_chunks[-1]["POS"] = morph_token.pos
                for feat in morph_token.feats.keys():
                    morph_chunks[-1][feat] = morph_token.feats[feat]
                counter += 1

        return pd.DataFrame(morph_chunks)
