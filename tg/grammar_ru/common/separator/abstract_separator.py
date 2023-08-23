from abc import ABC, abstractmethod
from typing import *

import pandas as pd

from .offsets import _generate_offsets
from ..data_bundle import DataBundle
from ..df_viewer import DfViewer


class AbstractSeparator(ABC):
    COLUMNS = ['word_id', 'sentence_id', 'word_index', 'paragraph_id', 'word_tail', 'word', 'word_type',
               'word_length']
    UPDATE_COLUMNS = ['updated', 'original_word_id', 'original_sentence_id', 'original_paragraph_id']

    @abstractmethod
    def _tokenize(self, text: str) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def _sentenize(self, text: str) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def _classify_word(self, word: str) -> str:
        raise NotImplementedError

    def _separate_string(self, s: str, word_id_start=0, sentence_id_start=0):
        result = []
        sentences = self._sentenize(s)
        offset = 0
        word_id = word_id_start
        sentence_id = sentence_id_start
        for sentence in sentences:
            tokens = [token for token in self._tokenize(sentence)]
            offsets, delta = _generate_offsets(s[offset:], tokens)
            for word_index, (word, word_tail) in enumerate(zip(tokens, offsets)):
                word_type = self._classify_word(word)
                result.append((word_id, sentence_id, word_index, word_tail, word, word_type))
                word_id += 1
            offset += delta
            sentence_id += 1
        df = pd.DataFrame(result, columns=['word_id', 'sentence_id', 'word_index', 'word_tail', 'word', 'word_type'])
        df['word_length'] = df.word.str.len()
        return df

    def separate_string(self, s: str):
        return self.separate_paragraphs(s.split('\n'))

    def separate_paragraphs(self, strings: List[str]):
        result = []
        word_id_start, sentence_id_start = 0, 0
        for i, s in enumerate(strings):
            df = self._separate_string(s, word_id_start, sentence_id_start)
            if df.shape[0] > 0:
                word_id_start = df.iloc[-1].word_id + 1
                sentence_id_start = df.iloc[-1].sentence_id + 1
            df['paragraph_id'] = i
            result.append(df)
        if len(result) > 0:
            rdf = pd.concat(result, ignore_index=True)
            rdf = rdf[AbstractSeparator.COLUMNS]
        else:
            rdf = pd.DataFrame({c: [] for c in AbstractSeparator.COLUMNS})
        return rdf

    def build_bundle(self, text: Union[str, List[str], pd.DataFrame, DataBundle], featurizers: Optional[List] = None):
        if isinstance(text, str):
            db = DataBundle(src=self.separate_string(text))
        elif isinstance(text, list):
            db = DataBundle(src=self.separate_paragraphs(text))
        elif isinstance(text, pd.DataFrame):
            db = DataBundle(src=text)
        elif isinstance(text, DataBundle):
            db = text
        else:
            raise ValueError(f'`text` must be str, DataFrame or DataBundle, but was {type(text)}')
        if featurizers is not None:
            for featurizer in featurizers:
                featurizer.featurize(db)
        return db

    @staticmethod
    def check_df(df):
        for c in AbstractSeparator.COLUMNS:
            if c not in df.columns:
                raise ValueError(
                    f"Column {c} is not in dataframe. Use a Separator output as an argument to avoid such problems.")

    INDEX_COLUMNS = ['word_id', 'sentence_id', 'paragraph_id']

    @staticmethod
    def reset_indices(df, offset=0, keep_originals=True) -> pd.DataFrame:
        if not isinstance(offset, int):
            raise ValueError(f'Offset must be `int`, but was: {offset}')
        df = df.copy()
        for column in AbstractSeparator.INDEX_COLUMNS:
            rc = df[column]
            rc = rc.drop_duplicates().sort_values().to_frame('value')
            rc = rc.reset_index(drop=True).reset_index(drop=False)
            rc.value = rc.value.astype(int)
            rc = rc.set_index('value')
            rc = rc['index']
            rc += offset
            if keep_originals:
                df['original_' + column] = df[column]
            df[column] = list(df[[column]].merge(rc, left_on=column, right_index=True)['index'])

        df.index = list(df.word_id)
        df['updated'] = False
        return df

    @staticmethod
    def get_max_id(df: pd.DataFrame) -> int:
        return int(df[AbstractSeparator.INDEX_COLUMNS].max().max()) + 2

    @staticmethod
    def validate(df: pd.DataFrame):
        for c in AbstractSeparator.INDEX_COLUMNS + ['word']:
            if c not in df.columns:
                raise ValueError(f'Column {c} is not found in dataframe')
        if df.shape[0] != df.drop_duplicates('word_id').shape[0]:
            non_unique = df.groupby('word_id').size().feed(lambda z: z.loc[z > 1]).index
            raise ValueError(f'Non-unique word id {list(non_unique)}')
        if df.word_id.isnull().any():
            raise ValueError(f'Null word_id at {list(df.loc[df.word_id.isnull()])}')

    @staticmethod
    def _from_word_en(words: Iterable[str], lang: str):
        df = pd.DataFrame(dict(word=list(words)))
        for c in AbstractSeparator.INDEX_COLUMNS:
            df[c] = list(range(df.shape[0]))
        df['word_type'] = lang
        df['word_tail'] = 1
        return df

    @abstractmethod
    def from_word_en(self, words: Iterable[str]) -> pd.DataFrame:
        pass

    Viewer = DfViewer
