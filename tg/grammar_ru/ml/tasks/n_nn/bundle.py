import typing as tp

import pandas as pd

from .word_normalizer import WordNormalizer, NltkWordStemmer


_single_n_regex = r'[^н]н[^н](?!.*?нн)'  # matches only 'н' not followed by 'нн'
_double_n_regex = r'нн(?!.+?н)'  # matches only 'нн' not followed by 'н'


def _extract_words(dfs: tp.Iterable[pd.DataFrame]) -> tp.Tuple[tp.Set[str], tp.Set[str]]:
    words_with_single_n = set()
    words_with_double_n = set()
    for df in dfs:
        words_with_single_n.update(df[df['word'].str.contains(_single_n_regex)]['word'])
        words_with_double_n.update(df[df['word'].str.contains(_double_n_regex)]['word'])

    return words_with_single_n, words_with_double_n


def transform_word_with_double_to_word_with_single(word: str) -> str:
    return word[::-1].replace('нн', 'н', 1)[::-1]


def build_dictionary(
        dfs: tp.Iterable[pd.DataFrame],
        normalizer: WordNormalizer = NltkWordStemmer()
        ) -> tp.Set[str]:
    words_with_single_n, words_with_double_n = _extract_words(dfs)

    words_with_double_replaced_to_single = set(map(transform_word_with_double_to_word_with_single, words_with_double_n))

    intersection = (
            set(map(normalizer.normalize_word, words_with_single_n))
            & set(map(normalizer.normalize_word, words_with_double_replaced_to_single))
    )

    return intersection
