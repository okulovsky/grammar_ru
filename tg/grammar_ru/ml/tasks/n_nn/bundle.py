import re
from collections import Counter
import typing as tp

import pandas as pd
import numpy as np

from tg.grammar_ru.algorithms import SpellcheckAlgorithm
from tg.grammar_ru.common import DataBundle
from .word_normalizer import WordNormalizer
from .regular_expressions import single_n_regex, double_n_regex


def _count_verbs(dfs: tp.Iterable[pd.DataFrame]) -> tp.Dict[str, int]:
    counter: tp.MutableMapping[str, int] = Counter()
    for df in dfs:
        suited = df[df['word'].str.contains(pat=single_n_regex) | df['word'].str.contains(pat=double_n_regex)]['word']
        for word in suited:
            counter[word.lower()] += 1

    return dict(counter)


def _another(word: str) -> str:
    if re.search(single_n_regex, word):
        return word[::-1].replace('н', 'нн', 1)[::-1]
    else:
        return word[::-1].replace('нн', 'н', 1)[::-1]


def _get_good_words_df(counted_words: tp.Dict[str, int]) -> pd.DataFrame:
    for word in list(counted_words):
        if _another(word) not in counted_words:
            counted_words[_another(word)] = 0

    rows = []
    for word in counted_words:
        if re.search(single_n_regex, word):
            rows.append(
                (word, _another(word), counted_words[word], counted_words[_another(word)])
                )

    df = pd.DataFrame(rows, columns=['n_word', 'nn_word', 'n_cnt', 'nn_cnt'])
    df['both_found'] = df[['n_cnt', 'nn_cnt']].min(axis=1) > 0
    df = df[df.both_found]
    df['ratio'] = np.minimum(df.n_cnt/df.nn_cnt, df.nn_cnt/df.n_cnt)

    alg = SpellcheckAlgorithm()
    for prefix in ['n_', 'nn_']:
        xdf = pd.DataFrame(dict(word=df[prefix+'word'], word_type='ru'))
        kdf = alg.run(DataBundle(src=xdf))
        df[prefix+'spell'] = ~kdf.error

    df['both_correct'] = df.n_spell & df.nn_spell
    df = df.loc[df.both_correct]
    return df


def build_dictionary(
        dfs: tp.Iterable[pd.DataFrame],
        normalizer: WordNormalizer
        ) -> tp.Set[str]:
    counted = _count_verbs(dfs)
    df = _get_good_words_df(counted)
    df['n_word'] = df['n_word'].apply(normalizer.normalize_word)
    df['nn_word'] = df['nn_word'].apply(normalizer.normalize_word)
    good_words = set(df.n_word).union(df.nn_word)

    return good_words
