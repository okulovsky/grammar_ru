from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple, Collection, Any

import pandas as pd

from .corpus_reader import CorpusReader
from .corpus_writer import CorpusWriter, CorpusFragment


class ParallelCorpus:
    def __init__(self, corpus_path: Path, subcorpus_name="subcorpus_name"):
        self.corpus_path = corpus_path
        self.subcorpus_name_column = subcorpus_name
        self.reader = CorpusReader(corpus_path)

    def get_mapped_data(self, uids: List[str], sub_corpus_types: Union[str, List[str]] = 'all') -> List[
        Dict[str, pd.DataFrame]]:
        relation_info = self.reader.read_relations()
        if sub_corpus_types == 'all':
            sub_corpus_types = self.reader.get_toc()[f'{self.subcorpus_name_column}'].unique()
        sub_corpuses_relations = [relation_info.loc[relation_info[self.subcorpus_name_column] == sub_name]
                                  for sub_name in sub_corpus_types]
        parallel_request = list()
        for uid in uids:
            relation_uid: Dict[str, str] = dict()
            for name, sub_corpus_relation in zip(sub_corpus_types, sub_corpuses_relations):
                relation_uid[name] = sub_corpus_relation.loc[sub_corpus_relation.file_1 == uid, 'file_2'].iloc[0]
            parallel_request.append(relation_uid)

        parallel_response: List[Dict[str, pd.DataFrame]] = self.reader.read_src(parallel_request)

        return parallel_response

    def __getattr__(self, item) -> CorpusReader:
        all_subcorpus_types = self.reader.get_toc()[f"{self.subcorpus_name_column}"].unique()
        if item not in all_subcorpus_types:
            raise AttributeError(f"{item} not found. If you want to get subcoprus, check available subtypes names : {all_subcorpus_types}")
        reader = CorpusReader(self.corpus_path)
        replaced_toc = reader.get_toc()
        replaced_toc = replaced_toc.loc[replaced_toc[f"{self.subcorpus_name_column}"] == item]
        reader.get_toc = lambda: replaced_toc
        reader.read_toc = lambda: replaced_toc
        return reader

    # def get_info(self, dict_of_needs: Dict[str, List[str]]) -> ParallelInfo:  #
    #     toc = CorpusReader(self.corpus_path).get_toc()
    #     info: Dict[str, pd.DataFrame] = dict()
    #     for subcorpus_type, ids in dict_of_needs.items():
    #         sub_toc: pd.DataFrame = toc[toc[self.subcorpus_name_column] == subcorpus_type]
    #         info[subcorpus_type] = sub_toc.loc[ids]
    #     return info
