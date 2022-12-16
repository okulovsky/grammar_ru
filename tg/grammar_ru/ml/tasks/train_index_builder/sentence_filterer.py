import abc
import typing as tp
from pathlib import Path

import pandas as pd

from tg.grammar_ru.ml.corpus import ITransfuseSelector
from tg.grammar_ru.ml.tasks.n_nn.word_normalizer import WordNormalizer


class SentenceFilterer(ITransfuseSelector):
    def __init__(self) -> None:
        self.ref_id = 0

    @abc.abstractmethod
    def get_targets(self, df: pd.DataFrame) -> pd.Series:
        pass

    def _get_good_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        df['is_target'] = self.get_targets(df)
        good_sentences = df.groupby('sentence_id').is_target.max().feed(lambda z: z.loc[z].index)

        return good_sentences

    def get_filtered_df(self, df: pd.DataFrame) -> pd.DataFrame:
        good_sentences = self._get_good_sentences(df)

        return df.loc[df.sentence_id.isin(good_sentences)].copy()

    def select(
            self,
            corpus: Path,
            df: pd.DataFrame,
            toc_row: tp.Dict
            ) -> tp.Union[tp.List[pd.DataFrame], pd.DataFrame]:
        return self.get_filtered_df(df)


class DictionaryFilterer(SentenceFilterer):
    def __init__(self, good_words: tp.Sequence[str]) -> None:
        super().__init__()
        self.good_words = good_words

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


class NormalizeFilterer(DictionaryFilterer):
    def __init__(self, good_words: tp.Sequence[str], word_normalizer: WordNormalizer) -> None:
        super().__init__(good_words=good_words)

        self._word_normalizer = word_normalizer

    def get_targets(self, df: pd.DataFrame) -> tp.Sequence[bool]:
        return df.word.apply(self._get_normalized_word).str.lower().isin(self.good_words)

    def _get_normalized_word(self, word: str) -> str:
        return self._word_normalizer.normalize_word(word)
