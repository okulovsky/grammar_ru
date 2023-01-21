from typing import *
import abc
import re
import itertools
import typing as tp

import pandas as pd
import numpy as np

from ...grammar_ru.common import Separator
from yo_fluq_ds import fluq

class NegativeSampler(abc.ABC):
    """Builds dataframe with negative samples from correct dataframe"""

    @abc.abstractmethod
    def build_negative_sample_from_positive(self, positive_sample: pd.DataFrame) -> pd.DataFrame:
        pass

    def build_all_negative_samples_from_positive(self, positive_sample: pd.DataFrame) -> List[pd.DataFrame]:
        results = []
        xdf = positive_sample
        qdf = xdf.loc[xdf.is_target]
        qdf = qdf.feed(fluq.add_ordering_column('sentence_id', 'word_id'))
        xdf = xdf.merge(qdf[['order']], left_index=True, right_index=True, how='left')
        xdf.order = xdf.order.fillna(-1).astype(int)
        orders = list(range(xdf.order.max()+1))
        for k in orders:
            if k == -1:
                continue
            kdf = xdf.copy()
            kdf.is_target = kdf.is_target & (kdf.order==k)
            kdf = kdf.drop('order',axis=1)
            good_sentences = kdf.loc[kdf.is_target].sentence_id
            kdf = kdf.loc[kdf.sentence_id.isin(good_sentences)]
            results.append(self.build_negative_sample_from_positive(kdf))
        return results


class EndingNegativeSampler(NegativeSampler):
    def __init__(self, ending_1, ending_2):
        self.ending_1 = ending_1
        self.ending_2 = ending_2


    def build_negative_sample_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
        negative = positive.copy()
        negative.word = np.where(
            ~negative.is_target,
            negative.word,
            np.where(
                negative.word.str.endswith(self.ending_1),
                negative.word.str.replace(self.ending_1, self.ending_2),
                negative.word.str.replace(self.ending_2, self.ending_1)
            )
        )
        return negative

class WordPairsNegativeSampler(NegativeSampler):
    def __init__(self, pairs: List[Tuple[str,str]]):
        self.pairs = pairs
        self.input_marker =  'ЫВЩЛЦЩУСЛФФЫВАЙЗДЫ'
        self.output_marker = 'ВЩЛЙДЫЗФЗСЧЯФБЦЬЛУ'


    def build_negative_sample_from_positive(self, positive_sample: pd.DataFrame) -> pd.DataFrame:
        df = positive_sample.copy()
        df.word = np.where(df.is_target, self.input_marker+df.word, df.word)
        text = Separator.Viewer().to_text(df)
        for p in self.pairs:
            for w_1, w_2 in [ (p[0], p[1]), (p[1], p[0])]:
                text = text.replace(self.input_marker+w_1, self.output_marker+w_2)
        ndf = Separator.separate_string(text)
        ndf['is_target'] = ndf.word.str.startswith(self.output_marker)
        ndf.word = np.where(ndf.word.str.startswith(self.output_marker), ndf.word.str.slice(len(self.output_marker)), ndf.word)
        return ndf




class RegexNegativeSampler(NegativeSampler):
    """Builds negative samples from sentences which contains given regexp"""

    def __init__(self, target_regex: str) -> None:
        self.target_regex = target_regex
        self._correct_word_suffix = '-корректноеслово'

    @abc.abstractmethod
    def get_alternatives(self, word: str) -> tp.List[str]:
        pass

    def _get_word_combinations(self, sentence: str) -> tp.Iterable[tp.Sequence[str]]:
        words = re.findall(self.target_regex, sentence, re.IGNORECASE)
        combinations = sorted(set(itertools.product(*(self.get_alternatives(word) for word in words))) - set([tuple(words)]))
        marked_alternatives = []
        for combination in combinations:
            marked = []
            for alternative, word in zip(combination, words):
                if alternative == word:
                    marked.append(f'{self._correct_word_suffix} '.join(word.split() + ['']).strip())
                else:
                    marked.append(alternative)
            marked_alternatives.append(marked)

        return marked_alternatives

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



