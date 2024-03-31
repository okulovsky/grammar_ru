from typing import *
from ..grammar_ru.algorithms import RepetitionsAlgorithm, SpellcheckAlgorithm, NlpAlgorithm
from ..grammar_ru.common import DataBundle, Loc
import pandas as pd
from yo_fluq_ds import FileIO

class PysmujAlgorithm(NlpAlgorithm):
    def __init__(self, tsa_algorithm: NlpAlgorithm):
        self.ignore_spellcheck_for_popular_words_borderline = 3
        self.repetitions_algorithm = RepetitionsAlgorithm(vicinity=20)
        self.spellcheck_algorithm = SpellcheckAlgorithm()
        self.tsa_algorithm = tsa_algorithm

    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        df = db.src
        if df.shape[0]==0:
            return None
        df = df.merge(db.pymorphy[['normal_form','POS']], left_on='word_id', right_index=True)
        df = df.merge(df.groupby('normal_form').size().to_frame('seen_in_text'), left_on='normal_form', right_index=True)

        dfs = []
        spellindex = (df.index.isin(index)) & (df.seen_in_text<self.ignore_spellcheck_for_popular_words_borderline)
        dfs.append(self.spellcheck_algorithm.run(db, spellindex))

        bad_pos = ['PREP','NPRO','NONE','INTJ','CONJ', 'PRCL']
        repindex = df.index.isin(index) & (~df.POS.isin(bad_pos))
        dfs.append(self.repetitions_algorithm.run(db, repindex))

        dfs.append(self.tsa_algorithm.run(db, index))
        return NlpAlgorithm.combine(db.src, dfs)

