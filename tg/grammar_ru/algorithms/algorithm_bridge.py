from typing import *
from ..algorithms import NlpAlgorithm
from ..features import PyMorphyFeaturizer
from ..common import DataBundle, Separator
import pandas as pd
import sys
import traceback

class BridgeData:
    def __init__(self,
                 result_df: Optional[pd.DataFrame],
                 old_paragraphs: List[str],
                 bundle: Optional[DataBundle],
                 check_index: pd.Index,
                 dumps: Dict
                 ):
        self.result_df = result_df
        self.old_paragraphs = old_paragraphs
        self.bundle = bundle
        self.check_index = check_index
        self.dumps = dumps


class AlgorithmBridge:
    def __init__(self, algorithm: NlpAlgorithm, debug = False):
        self.algorithm = algorithm
        self.pymorphy = PyMorphyFeaturizer()
        self.debug = debug

    def run(self,
            paragraphs: List[str],
            paragraph_to_old_index: Optional[List[Optional[int]]] = None,
            old_data: Optional[BridgeData] = None):

        def _build_dump(self, *args):
            ex = sys.exc_info()
            return dict(
                type=str(ex[0]),
                value=str(ex[1]),
                trace=traceback.format_exception(*ex),
                main_args = [paragraphs, paragraph_to_old_index, old_data]
            )

        dumps = {}
        if paragraph_to_old_index is None or old_data is None or old_data.bundle is None:
            bundle = Separator.build_bundle(paragraphs, [self.pymorphy])
        else:
            try:
                bundle = Separator.update_bundle(old_data.bundle, paragraphs, paragraph_to_old_index, [self.pymorphy])
            except:
                dumps['updating_bundle_error'] = _build_dump(old_data.bundle.src, paragraphs, paragraph_to_old_index)
                bundle = Separator.build_bundle(paragraphs, [self.pymorphy])
                if self.debug:
                    raise
        if bundle.src.shape[0] == 0:
            return BridgeData(None, paragraphs, None, None, {})
        changed_paragraphs = [i for i in range(len(paragraphs)) if paragraph_to_old_index is None or paragraph_to_old_index[i] is None]
        index = bundle.src.loc[bundle.src.paragraph_id.isin(changed_paragraphs)].index
        try:
            result_df = self.algorithm.run(bundle, index)
            result_df = bundle.src.merge(result_df, left_on='word_id', right_index=True)
        except:
            result_df = None
            dumps['algorithm_error'] = _build_dump(bundle, index)
            if self.debug:
                raise
        return BridgeData(result_df, paragraphs, bundle, index, dumps)

        





