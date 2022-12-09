import typing as tp
import abc

import pandas as pd

from tg.common.ml.batched_training import train_display_test_split
from tg.grammar_ru.ml.tasks.train_index_builder.sentence_filterer import SentenceFilterer
from tg.grammar_ru.ml.tasks.train_index_builder.negative_sampler import NegativeSampler


class IndexBuilder(abc.ABC):
    def __init__(
            self,
            filterer: SentenceFilterer,
            negative_sampler: NegativeSampler,
            add_negative_samples: bool = True
            ) -> None:
        self.ref_id = 0
        self.add_negative_samples = add_negative_samples
        self.filterer = filterer
        self.negative_sampler = negative_sampler

    def build_train_index(self, df: pd.DataFrame) -> tp.List[pd.DataFrame]:
        ddf = df.iloc[[0]]
        description = (ddf.corpus_id+'/'+ddf.file_id).iloc[0]
        df['original_corpus_id'] = df.corpus_id

        filtered = self.filterer.filter_sentences(df=df)

        index = [filtered]
        if self.add_negative_samples:
            negative = self.negative_sampler.build_negative_sample_from_positive(filtered)
            negative['is_target'] = self.filterer.get_targets(negative)
            index.append(negative)

        for frame in index:
            if frame.sentence_id.isnull().any():
                raise ValueError(f"Null sentence id when processing, uid {description}")

        return index

    @staticmethod
    def build_index_from_src(src_df: pd.DataFrame) -> pd.DataFrame:
        df = src_df.loc[src_df.is_target][['word_id', 'sentence_id', 'label', 'reference_sentence_id']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
