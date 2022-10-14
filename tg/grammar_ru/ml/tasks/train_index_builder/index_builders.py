import typing as tp
import re

import numpy as np
import pandas as pd

from .train_index_builder import DictionaryIndexBuilder
from ..n_nn.word_normalizer import WordNormalizer, NltkWordStemmer


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
            word_normalizer: WordNormalizer = NltkWordStemmer(),
            add_negative_samples: bool = True):
        super().__init__(
            good_words=good_words,
            add_negative_samples=add_negative_samples)
        self._word_normalizer = word_normalizer
        self._double_n_regex = r'нн(?!.+?н)'  # matches only 'нн' not followed by 'н'

    def _get_targets(self, df: pd.DataFrame) -> pd.Series:
        return df.word.apply(self._get_normalized_word).str.lower().isin(self.good_words)

    def _build_negative_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
        negative = positive.copy()
        negative.word = np.where(
            ~negative.is_target,
            negative.word,
            np.where(
                negative.word.str.contains(r'[^н]н[^н](?!.*?нн)'),
                negative.word.str[::-1].str.replace('н', 'нн', 1).str[::-1],
                negative.word.str[::-1].str.replace('нн', 'н', 1).str[::-1]
            )
        )

        negative['label'] = 1

        return negative

    def _get_normalized_word(self, word: str) -> str:
        if re.match(self._double_n_regex, word):
            word = word[::-1].replace('нн', 'н', 1)[::-1]

        return self._word_normalizer.normalize_word(word)
