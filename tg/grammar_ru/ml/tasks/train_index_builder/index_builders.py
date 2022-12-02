import typing as tp

import numpy as np
import pandas as pd

from .train_index_builder import DictionaryIndexBuilder
from ..n_nn.word_normalizer import WordNormalizer
from ..n_nn.regular_expressions import single_n_regex
from ....common import Separator


class TsaIndexBuilder(DictionaryIndexBuilder):
    def __init__(self, good_words: tp.Sequence[str], add_negative_samples: bool = True) -> None:
        self.good_words = good_words
        super().__init__(add_negative_samples=add_negative_samples)

    def _get_targets(self, df: pd.DataFrame) -> pd.Series:
        return df.word.str.lower().isin(self.good_words)

    def _build_negative_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
        negative = positive.copy()
        negative.word = np.where(
            ~negative.is_target,
            negative.word,
            np.where(
                negative.word.str.endswith('тся'),
                negative.word.str.replace('тся', 'ться'),
                negative.word.str.replace('ться', 'тся')
            )
        )
        negative['label'] = 1

        return negative


class NNnIndexBuilder(DictionaryIndexBuilder):
    def __init__(
            self,
            good_words: tp.Sequence[str],
            word_normalizer: WordNormalizer,
            add_negative_samples: bool = True):
        super().__init__(
            add_negative_samples=add_negative_samples)
        self.good_words = good_words
        self._word_normalizer = word_normalizer

    def _get_targets(self, df: pd.DataFrame) -> pd.Series:
        return df.word.apply(self._get_normalized_word).str.lower().isin(self.good_words)

    def _build_negative_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
        negative = positive.copy()
        negative.word = np.where(
            ~negative.is_target,
            negative.word,
            np.where(
                negative.word.str.contains(single_n_regex),
                negative.word.str[::-1].str.replace('н', 'нн', 1).str[::-1],
                negative.word.str[::-1].str.replace('нн', 'н', 1).str[::-1]
            )
        )

        negative['label'] = 1

        return negative

    def _get_normalized_word(self, word: str) -> str:
        return self._word_normalizer.normalize_word(word)


class ChtobyIndexBuilder(DictionaryIndexBuilder):
    def __init__(self, add_negative_samples: bool = True):
        super().__init__(add_negative_samples=add_negative_samples)
        self.good_words = ['чтобы', 'что бы']

    def _get_targets(self, df: pd.DataFrame) -> tp.Sequence[bool]:
        targets = df.set_index('word_id').word.str.lower() == 'чтобы'

        chto = df[df['word'] == 'что']
        chto_next = chto['word_id'] + 1
        chto_neighbour = df.merge(chto_next, how='inner')
        by = chto_neighbour[chto_neighbour['word'] == 'бы']

        targets[by['word_id'] - 1] = True

        return targets.values

    def _build_negative_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
        negative = positive.copy()
        chtoby = negative['word'] == 'чтобы'

        # transforming 'что' + 'бы' to 'чтобы'
        chto = negative[negative['word'] == 'что']
        chto_next = chto[['sentence_id', 'word_index']].copy()
        chto_next['word_index'] += 1
        chto_neighbor = negative.merge(chto_next, on=['sentence_id', 'word_index'], how='inner')
        by = chto_neighbor[chto_neighbor['word'] == 'бы']

        if by.shape[0]:
            chto_with_pair_loc = by['word_id']
            negative.loc[chto_with_pair_loc - 1, 'word'] = 'чтобы'

        # transforming 'чтобы' to 'что' + 'бы'
        negative.loc[chtoby, 'word'] = 'что бы'
        if by.shape[0]:
            negative = negative.drop(chto_with_pair_loc)
        negative = Separator.separate_string(Separator.Viewer().to_text(negative))
        negative['is_target'] = self._get_targets(negative)

        negative['label'] = 1

        return negative
