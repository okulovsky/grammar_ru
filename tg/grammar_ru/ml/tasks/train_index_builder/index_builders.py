import typing as tp

import numpy as np
import pandas as pd

from .train_index_builder import DictionaryIndexBuilder
from ..n_nn.word_normalizer import WordNormalizer
from ..n_nn.regular_expressions import single_n_regex


class TsaIndexBuilder(DictionaryIndexBuilder):
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

    def _get_targets(self, df: pd.DataFrame) -> pd.Series:
        return df.word.str.lower().isin(self.good_words)

    def _get_another(self, word: str) -> str:
        if word == 'чтобы':
            return 'что бы'
        elif word == 'что бы':
            return 'чтобы'

        return word

    def _build_negative_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
        negative = positive.copy()
        negative.word = np.where(
            ~negative.is_target,
            negative.word,
            np.where(
                negative['word'].apply(lambda word: word in ('чтобы', 'что бы')),
                negative['word'].apply(self._get_another),
                negative['word'].apply(self._get_another)
            )
        )

        negative['label'] = 1

        return negative

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        # transforming 'что' + 'бы' to 'что бы'
        result = df.copy()
        what = result[result['word'] == 'что']
        what_next = what[['sentence_id', 'word_index']].copy()
        what_next['word_index'] += 1
        what_neighbour = result.merge(what_next, on=['sentence_id', 'word_index'], how='inner')
        would = what_neighbour[what_neighbour['word'] == 'бы']

        if would.shape[0] != 0:
            what_with_pair_loc = would['word_id']
            result.loc[what_with_pair_loc - 1, 'word'] = 'что бы'
            result.loc[what_with_pair_loc - 1, 'word_length'] += 3

            # word_index now is inconsistent
            return result.drop(what_with_pair_loc)
