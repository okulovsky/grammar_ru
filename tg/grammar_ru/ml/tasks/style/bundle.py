from tg.grammar_ru.ml.corpus.transfuse_selector import ITransfuseSelector
from tg.common.ml.batched_training import train_display_test_split
from typing import Union, Dict, List
from pathlib import Path
import pandas as pd


class AuthorSelector(ITransfuseSelector):
    def __init__(self, authors: List[str], author_column_name: str = "author") -> None:
        if isinstance(authors, str):
            authors = set(authors)
        self.authors = authors
        self.author_column_name = author_column_name
        
    def select(
        self, corpus: Path, df: pd.DataFrame, toc_row: Dict) -> Union[List[pd.DataFrame], pd.DataFrame]:
        
        if "original_corpus_id" not in df.columns:
            df["original_corpus_id"] = df.corpus_id

        author = toc_row[self.author_column_name]
        if author not in self.authors:
            return []

        df["author"] = author
        return df


class StyleIndexBuilder(ITransfuseSelector):
    def __init__(self) -> None:
        self.corpus_labels = dict()

    def select(
        self, corpus: Path, df: pd.DataFrame, toc_row: Dict) -> Union[List[pd.DataFrame], pd.DataFrame]:
        for label in set(df.original_corpus_id.values):
            if label not in self.corpus_labels:
                self.corpus_labels[label] = len(self.corpus_labels)

        target_word_ids = df[df.word_index == 0].word_id
        df["is_target"] = df.word_id.isin(target_word_ids)
        df["label"] = df.original_corpus_id.apply(lambda x: self.corpus_labels[x])

        return df

    @staticmethod
    def build_index_from_src(src_df: pd.DataFrame) -> pd.DataFrame:
        df = src_df.loc[src_df.is_target][['word_id', 'sentence_id', 'paragraph_id', 'label']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
