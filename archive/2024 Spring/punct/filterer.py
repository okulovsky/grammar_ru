import typing as tp

import pandas as pd

from tg.projects.alternative.sentence_filterer import SentenceFilterer


class PunctFilterer(SentenceFilterer):
    def __init__(self, symbols: tp.List[str]):
        super().__init__()
        self.symbols = symbols

    def get_targets(self, df: pd.DataFrame) -> pd.Series:
        return (
            df.word.shift(-1).isin(self.symbols)
            & (df.sentence_id == df.sentence_id.shift(-1))
            & (df.word_type != 'punct')
            )
