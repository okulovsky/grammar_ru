from dataclasses import dataclass
from typing import *
import pandas as pd
from grammar_ru.common import Separator, DataBundle
import numpy as np

@dataclass
class NlpErrorInfo:
    index: Optional[int]
    word: Optional[str]
    error: Optional[bool]
    suggest: Optional[str]
    algorithm: Optional[str]
    hint: Optional[str]
    error_type: Optional[str]

class NlpAlgorithm:
    Error = 'error'
    Suggest = 'suggest'
    Algorithm = 'algorithm'
    Hint = 'hint'
    ErrorType = 'error_type'

    class ErrorTypes:
        Unknown = 'unknown'
        Orthographic = 'orthographic'
        Grammatic = 'grammatic'
        Stylistic = 'stylistic'

    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        raise NotImplementedError()

    @staticmethod
    def _post_check(src, rdf, type_name):
        rdf = src[[]].merge(rdf, left_index=True, right_index=True, how='left')
        rdf[NlpAlgorithm.Error] = rdf[NlpAlgorithm.Error].fillna(False)
        defaults = {
            NlpAlgorithm.Algorithm: type_name,
            NlpAlgorithm.Hint: None,
            NlpAlgorithm.ErrorType: NlpAlgorithm.ErrorTypes.Unknown,
            NlpAlgorithm.Suggest: None
        }
        for key, value in defaults.items():
            if key not in rdf.columns:
                rdf[key] = value
            rdf[key] = np.where(rdf[NlpAlgorithm.Error], rdf[key], None)

        accepted_columns = list(defaults)+[NlpAlgorithm.Error]
        for c in list(rdf.columns):
            if c not in accepted_columns:
                rdf = rdf.drop(c, axis=1)

        return rdf


    def run(self, db: DataBundle, index: Optional[pd.Index] = None) -> pd.DataFrame:
        if index is None:
            index = db.src.index
        rdf = self._run_inner(db, index)
        if rdf is None or rdf.shape[0] == 0:
            rdf = pd.DataFrame({NlpAlgorithm.Error: False}, index=[])
        if NlpAlgorithm.Error not in rdf.columns:
            raise ValueError(f"{type(self)} must set {NlpAlgorithm.Error} column")
        return NlpAlgorithm._post_check(db.src, rdf, type(self).__name__)

    def run_on_string(self, s, index = None):
        db = Separator.build_bundle(s)
        return self.run(db, db.src.index if index is None else db.src.loc[db.src.index.isin(index)].index)

    def new_run(self, db: DataBundle, index: Optional[pd.Index] = None) -> List[NlpErrorInfo]:
        old_run_df = self.run(db, index)

        # List to store NlpErrorInfo objects
        error_info_list = []

        for _, row in old_run_df.iterrows():
            word = db.src.loc[row.name, 'word']
            index = db.src.loc[row.name, 'word_id']
            error_info = NlpErrorInfo(
                index=int(index),
                word=word,
                error=row.get(NlpAlgorithm.Error, None),
                suggest=row.get(NlpAlgorithm.Suggest, None),
                algorithm=row.get(NlpAlgorithm.Algorithm, None),
                hint=row.get(NlpAlgorithm.Hint, None),
                error_type=row.get(NlpAlgorithm.ErrorType, None)
            )
            if error_info.error:
                error_info_list.append(error_info)

        return error_info_list

    def new_run_on_string(self, s, index = None):
        db = Separator.build_bundle(s)
        return self.new_run(db, db.src.index if index is None else db.src.loc[db.src.index.isin(index)].index)

    def new_run_on_string_multiple_algorithms(self, s, algorithms: List):
        error_info_list = []
        for algorithm in algorithms:
            error_info_list.extend(algorithm.new_run_on_string(s))
        error_info_list = sorted(error_info_list, key=lambda x: x.index)
        return error_info_list

    @staticmethod
    def combine(src: pd.DataFrame, check_dfs: List[pd.DataFrame]):
        rdf = pd.DataFrame({
            'active' : True,
            NlpAlgorithm.Error : False,
            NlpAlgorithm.Algorithm : None,
            NlpAlgorithm.Suggest : None,
            NlpAlgorithm.Hint : None,
        }, index = src.index)


        for cdf in check_dfs:
            cdf = rdf[[]].merge(cdf, left_index=True, right_index=True, how='left')
            cdf[NlpAlgorithm.Error] = cdf[NlpAlgorithm.Error].fillna(False)
            rdf.loc[rdf.active, NlpAlgorithm.Error] = cdf.loc[rdf.active, NlpAlgorithm.Error]
            current = rdf.active & cdf[NlpAlgorithm.Error]

            for column in [NlpAlgorithm.Suggest, NlpAlgorithm.Algorithm, NlpAlgorithm.Hint, NlpAlgorithm.ErrorType]:
                rdf.loc[current, column] = cdf.loc[current, column]
            rdf.active = rdf.active & ~cdf[NlpAlgorithm.Error]

        rdf = rdf.drop('active', axis=1)
        return NlpAlgorithm._post_check(src, rdf, None)

    @staticmethod
    def new_combine_algorithms(db: DataBundle, index: pd.Index, *algorithms):
        dfs = [a.new_run(db,index) for a in algorithms]
        return dfs

    @staticmethod
    def combine_algorithms(db: DataBundle, index: pd.Index, *algorithms):
        dfs = [a.run(db,index) for a in algorithms]
        return NlpAlgorithm.combine(db.src, dfs)


