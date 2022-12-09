import abc
import typing as tp

import pandas as pd

from tg.grammar_ru.ml.tasks.n_nn.word_normalizer import WordNormalizer


class SentenceFilterer(abc.ABC):
    def __init__(self) -> None:
        self.ref_id = 0

    @abc.abstractmethod
    def get_targets(self, df: pd.DataFrame) -> pd.Series:
        pass

    def filter_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        df['is_target'] = self.get_targets(df)

        good_sentences = df.groupby('sentence_id').is_target.max().feed(lambda z: z.loc[z].index)
        filtered = df.loc[df.sentence_id.isin(good_sentences)].copy()
        filtered['label'] = 0

        ref_map = {v: self.ref_id + k for k, v in enumerate(filtered.sentence_id.unique())}
        filtered['reference_sentence_id'] = filtered.sentence_id.replace(ref_map)
        self.ref_id += 1 + len(filtered.sentence_id.unique())

        return filtered


class DictionaryFilterer(SentenceFilterer):
    def __init__(self, good_words: tp.Sequence[str]) -> None:
        super().__init__()
        self.good_words = good_words

    def get_targets(self, df: pd.DataFrame) -> pd.Series:
        return df.word.str.lower().isin(self.good_words)


class ChtobyFilterer(SentenceFilterer):
    def get_targets(self, df: pd.DataFrame) -> tp.Sequence[bool]:
        targets = df.set_index('word_id').word.str.lower() == 'чтобы'

        chto = df[df['word'] == 'что']
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
