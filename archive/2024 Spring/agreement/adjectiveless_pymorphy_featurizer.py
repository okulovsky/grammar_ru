from tg.common._common.data_bundle import DataBundle
from tg.grammar_ru.features.pymorphy_featurizer import PyMorphyFeaturizer
import numpy as np

class AdjectivelessPyMorphyFeaturizer(PyMorphyFeaturizer):
    def __init__(self):
        super().__init__()

    def _featurize_inner(self, db: DataBundle):
        df = super()._featurize_inner(db)
        adjectives = df[(df.POS.isin({"ADJF", "ADJS"}))].copy()  # TODO what else?
        nullable_cols = list(set(df.columns) - {'word_id', 'normal_form',
                                           'alternatives', 'score', 'delta_score','POS'})
        adjectives[nullable_cols] = np.nan
        df.loc[adjectives.index] = adjectives        
        return df

        