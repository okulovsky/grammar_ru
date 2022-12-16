import abc
import re
import itertools
import typing as tp

import pandas as pd
import numpy as np

from tg.grammar_ru.common import Separator
from tg.grammar_ru.ml.tasks.n_nn.regular_expressions import single_n_regex


class NegativeSampler(abc.ABC):
    """Builds dataframe with negative samples from correct dataframe"""

    @abc.abstractmethod
    def build_negative_sample_from_positive(self, positive_sample: pd.DataFrame) -> pd.DataFrame:
        pass


class RegexNegativeSampler(NegativeSampler):
    """Builds negative samples from sentences which contains given regexp"""

    def __init__(self, target_regex: str) -> None:
        self.target_regex = target_regex

    @abc.abstractmethod
    def get_alternatives(self, word: str) -> tp.List[str]:
        pass

    def _get_word_combinations(self, sentence: str) -> tp.Iterable[tp.Sequence[str]]:
        words = re.findall(self.target_regex, sentence, re.IGNORECASE)

        return set(itertools.product(*(self.get_alternatives(word) for word in words))) - set([tuple(words)])

    def _join_sentence_parts(
            self,
            sentence_parts: tp.Sequence[str],
            words_combination: tp.Sequence[str],
            starts_with_target: bool
            ) -> str:
        rebuilt_sentence = [''] * (len(words_combination) + len(sentence_parts))
        if starts_with_target:
            rebuilt_sentence[::2] = words_combination
            rebuilt_sentence[1::2] = sentence_parts
        else:
            rebuilt_sentence[1::2] = words_combination
            rebuilt_sentence[::2] = sentence_parts

        return ''.join(rebuilt_sentence)

    def _build_negative_sentences(self, sentence: str) -> tp.List[str]:
        sentence_parts = re.split(self.target_regex, sentence)
        starts_with_target = sentence_parts[0] == ''
        sentence_parts = list(filter(lambda word: word != '', sentence_parts))
        negative_samples = []
        for combination in self._get_word_combinations(sentence):
            negative_samples.append(self._join_sentence_parts(sentence_parts, combination, starts_with_target))

        return negative_samples

    def build_negative_sample_from_positive(self, positive_sample: pd.DataFrame) -> pd.DataFrame:
        sentences_df = (  # grouping all setnences
            positive_sample
            .assign(word_print=(
                positive_sample.word + pd.Series(' ', index=positive_sample.index) * positive_sample.word_tail
            ))
            .groupby('sentence_id').word_print
            .sum()
        )
        negative = sentences_df.apply(self._build_negative_sentences).explode()
        # TODO: mark correct words with target=0
        negative = Separator.separate_string(negative.str.cat(sep=' '))

        negative['label'] = 1

        return negative


class ChtobyNegativeSampler(RegexNegativeSampler):
    def __init__(self) -> None:
        super().__init__(target_regex=r'[чЧ]то бы|[чЧ]тобы')

    def get_alternatives(self, word: str) -> tp.List[str]:
        if word.lower() not in ('чтобы', 'что бы'):
            raise ValueError(f'got unexpected word: {word}')

        alternatives = [word]
        # saving first letter register
        if word.lower() == 'чтобы':
            alternatives.append(word[0] + 'то бы')
        if word.lower() == 'что бы':
            alternatives.append(word[0] + 'тобы')

        return alternatives


class TsaNegativeSampler(NegativeSampler):
    def build_negative_sample_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
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


class NNnNegativeSampler(NegativeSampler):
    def build_negative_sample_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
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
