import typing as tp
from pathlib import Path

import pandas as pd

from tg.common.ml.batched_training import train_display_test_split
from tg.grammar_ru.ml.corpus import ITransfuseSelector
from tg.grammar_ru.ml.tasks.train_index_builder.sentence_filterer import SentenceFilterer
from tg.grammar_ru.ml.tasks.train_index_builder.negative_sampler import NegativeSampler


class IndexBuilder(ITransfuseSelector):
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

    def build_train_index(self, filtered: pd.DataFrame) -> tp.List[pd.DataFrame]:
        ddf = filtered.iloc[[0]]
        description = (ddf.corpus_id+'/'+ddf.file_id).iloc[0]
        filtered['original_corpus_id'] = filtered.corpus_id
        filtered['label'] = 0

        ref_map = {v: self.ref_id + k for k, v in enumerate(filtered.sentence_id.unique())}
        filtered['reference_sentence_id'] = filtered.sentence_id.replace(ref_map)
        self.ref_id += 1 + len(filtered.sentence_id.unique())

        index = [filtered]
        if self.add_negative_samples:
            negative = self.negative_sampler.build_negative_sample_from_positive(filtered)
            negative['is_target'] = self.filterer.get_targets(negative)
            # TODO: move marking with target to negative sampler
            index.append(negative)

        for frame in index:
            if frame.sentence_id.isnull().any():
                raise ValueError(f"Null sentence id when processing, uid {description}")

        return index

    def select(
            self,
            path: Path,
            df: pd.DataFrame,
            toc_row: tp.Dict
            ) -> tp.Union[tp.List[pd.DataFrame], pd.DataFrame]:
        return self.build_train_index(df)

    @staticmethod
    def build_index_from_src(src_df: pd.DataFrame) -> pd.DataFrame:
        df = src_df.loc[src_df.is_target][['word_id', 'sentence_id', 'label', 'reference_sentence_id']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
