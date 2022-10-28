import typing as tp

import pandas as pd

from .word_normalizer import WordNormalizer
from .regular_expressions import single_n_regex, double_n_regex


def _extract_words(dfs: tp.Iterable[pd.DataFrame]) -> tp.Tuple[tp.Set[str], tp.Set[str]]:
    words_with_single_n = set()
    words_with_double_n = set()
    for df in dfs:
        words_with_single_n.update(df[df['word'].str.contains(single_n_regex)]['word'])
        words_with_double_n.update(df[df['word'].str.contains(double_n_regex)]['word'])

    return words_with_single_n, words_with_double_n


def build_dictionary(
        dfs: tp.Iterable[pd.DataFrame],
        normalizer: WordNormalizer
        ) -> tp.Set[str]:
    words_with_single_n, words_with_double_n = _extract_words(dfs)

    intersection = (
            set(map(normalizer.normalize_word, words_with_single_n))
            & set(map(normalizer.normalize_word, words_with_double_n))
    )

    return intersection
