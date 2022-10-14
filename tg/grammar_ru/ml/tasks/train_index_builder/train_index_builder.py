import typing as tp
import abc

import pandas as pd

from tg.common.ml.batched_training import train_display_test_split


class DictionaryIndexBuilder(abc.ABC):
    def __init__(self, good_words: tp.Sequence[str], add_negative_samples: bool = True):
        self.good_words = good_words
        self.ref_id = 0
        self.add_negative_samples = add_negative_samples

    def build_train_index(self, df: pd.DataFrame) -> tp.List[pd.DataFrame]:
        ddf = df.iloc[[0]]
        description = (ddf.corpus_id+'/'+ddf.file_id).iloc[0]

        df['original_corpus_id'] = df.corpus_id
        df['is_target'] = self._get_targets(df)

        positive = self._build_positive(df)
        ar = [positive]
        if self.add_negative_samples:
            ar.append(self._build_negative_from_positive(positive))

        for f in ar:
            if f.sentence_id.isnull().any():
                raise ValueError(f"Null sentence id when processing, uid {description}")

        return ar

    @abc.abstractmethod
    def _get_targets(self, df: pd.DataFrame) -> pd.Series:
        pass

    def _build_positive(self, df: pd.DataFrame) -> pd.DataFrame:
        good_sentences = df.groupby('sentence_id').is_target.max().feed(lambda z: z.loc[z].index)
        positive = df.loc[df.sentence_id.isin(good_sentences)].copy()
        positive['label'] = 0

        ref_map = {v: self.ref_id + k for k, v in enumerate(positive.sentence_id.unique())}
        positive['reference_sentence_id'] = positive.sentence_id.replace(ref_map)
        self.ref_id += 1 + len(positive.sentence_id.unique())

        return positive

    @abc.abstractmethod
    def _build_negative_from_positive(self, positive: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def build_index_from_src(src_df: pd.DataFrame) -> pd.DataFrame:
        df = src_df.loc[src_df.is_target][['word_id', 'sentence_id', 'label', 'reference_sentence_id']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
