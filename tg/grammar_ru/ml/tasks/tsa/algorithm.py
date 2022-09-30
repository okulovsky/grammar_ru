from typing import *
from ....algorithms import NlpAlgorithm
from ....common import DataBundle, Separator
from .....common.ml.batched_training import IndexedDataBundle, BatchedTrainingTask
from ...features import *
import pandas as pd
import numpy as np

class TsaAlgorithm(NlpAlgorithm):
    def __init__(self,
                 model: BatchedTrainingTask,
                 words,
                 borderline,
                 only_pymotphy_required
                 ):
        self.model = model
        self.words = words
        self.borderline = borderline
        self.only_pymorphy_required = only_pymotphy_required
        self.featurizers = None

    def _lazy_init(self):
        if self.featurizers is None:
            if self.only_pymorphy_required:
                self.featurizers = [
                    PyMorphyFeaturizer(),
                ]
            else:
                self.featurizers = [
                    PyMorphyFeaturizer(),
                    SlovnetFeaturizer(),
                    SyntaxTreeFeaturizer(),
                    SyntaxStatsFeaturizer()
                ]

    def _build_prediction_frame(self, df):
        self._lazy_init()
        idf = df.loc[df.word.str.lower().isin(self.words)].copy()
        idf['label'] = 0
        idf.index.name='sample_id'
        if idf.shape[0]==0:
            return None
        df = df.loc[df.sentence_id.isin(idf.sentence_id)]
        bundle = Separator.build_bundle(df, self.featurizers)
        ibundle = IndexedDataBundle(idf, bundle)

        result = self.model.predict(ibundle)
        idf = idf.merge(result[['predicted']], left_index=True, right_index=True)
        return idf


    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        df = db.src.loc[index]
        idf = self._build_prediction_frame(df)
        if idf is None:
            return None
        idf = idf.loc[idf.predicted>self.borderline]
        if idf.shape[0] == 0:
            return None
        idf[NlpAlgorithm.Error] = True
        idf[NlpAlgorithm.Algorithm] = 'tsa'
        idf[NlpAlgorithm.ErrorType] = NlpAlgorithm.ErrorTypes.Grammatic
        idf[NlpAlgorithm.Hint] = (100*idf.predicted).astype(int).astype(str)
        idf[NlpAlgorithm.Suggest] = np.where(
            idf.word.str.endswith('тся'),
            idf.word.str.replace('тся', 'ться'),
            idf.word.str.replace('ться', 'тся')
        )
        return idf


