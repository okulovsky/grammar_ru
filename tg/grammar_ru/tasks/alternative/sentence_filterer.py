import abc
import typing as tp
from pathlib import Path
import pandas as pd
import deprecated


class SentenceFilterer:
    """Filters sentences from given dataframe"""

    def __init__(self) -> None:
        self.ref_id = 0

    @abc.abstractmethod
    def get_targets(self, df: pd.DataFrame) -> pd.Series:
        pass

    def _get_good_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        df['is_target'] = self.get_targets(df)
        good_sentences = df.groupby('sentence_id').is_target.max().feed(lambda z: z.loc[z].index)

        return good_sentences

    @deprecated.deprecated('Use `filter` instead')
    def get_filtered_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.filter(df)

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        good_sentences = self._get_good_sentences(df)
        return df.loc[df.sentence_id.isin(good_sentences)].copy()




class DictionaryFilterer(SentenceFilterer):
    """Filters sentences which contains given words"""

    def __init__(self, good_words: tp.Sequence[str]) -> None:
        super().__init__()
        self.good_words = set(good_words)

    def get_targets(self, df: pd.DataFrame) -> pd.Series:
        return df.word.str.lower().isin(self.good_words)


class ChtobyFilterer(SentenceFilterer):
    def get_targets(self, df: pd.DataFrame) -> tp.Sequence[bool]:
        targets = df.set_index('word_id').word.str.lower() == 'чтобы'

        chto = df[df['word'].str.lower() == 'что']
        chto_next = chto['word_id'] + 1
        chto_neighbour = df.merge(chto_next, how='inner')
        by = chto_neighbour[chto_neighbour['word'] == 'бы']

        targets[by['word_id'] - 1] = True

        return targets.values

