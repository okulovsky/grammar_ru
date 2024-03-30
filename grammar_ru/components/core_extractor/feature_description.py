from typing import *
from ....common.ml import dft
import numpy as np

def log_transformer(df):
    return np.log(df+1)


def cont_transformer(df):
    return df.astype(float)



class FeatureSelection:
    def __init__(self,
                 from_frame: str,
                 continuous: Optional[List[str]] = None,
                 continuous_logarithmic: Optional[List[str]] = None,
                 categorical: Optional[List[str]] = None,
                 ):
        self.from_frame = from_frame
        self.continuous = continuous
        self.continuous_logarithmic = continuous_logarithmic
        self.categorical = categorical

    def to_transformer(self):
        transformers = []
        if self.continuous is not None:
            transformers.append(dft.ContinousTransformer(self.continuous, preprocessor=cont_transformer))
        if self.continuous_logarithmic is not None:
            transformers.append(dft.ContinousTransformer(self.continuous_logarithmic, preprocessor=log_transformer))
        if self.categorical is not None:
            transformers.append(dft.CategoricalTransformer2(self.categorical, 25))
        return dft.DataFrameTransformer(transformers)




features = {}

features['pymorphy'] = FeatureSelection(
    'pymorphy',
    continuous = [
        'score',
        'delta_score',
    ],
    continuous_logarithmic= [
        'alternatives'
    ],
    categorical= [
        'POS',
        'animacy',
        'gender',
        'number',
        'case',
        'aspect',
        'transitivity',
        'person',
        'tense',
        'mood',
        'voice',
        'involvement'
    ]
)

features['slovnet_morph'] = FeatureSelection(
    'slovnet',
    categorical= [
        'POS',
        'Animacy',
         'Case',
         'Gender',
         'Number',
         'Aspect',
         'Mood',
         'Person',
         'Tense',
         'VerbForm',
         'Voice',
         'Degree',
         'Foreign',
         'Variant',
         'Polarity',
         'Hyph'
])

features['slovnet_syntax'] = FeatureSelection(
        'slovnet',
        categorical=[
            'relation'
        ]
)

features['syntax_fixes'] = FeatureSelection(
        'syntax_fixes',
        categorical=[
            'root',
            'cycle_status',
        ]
)

features['syntax_stats'] = FeatureSelection(
    'syntax_stats',
    continuous_logarithmic = [
     'children',
     'descendants',
     'sentence_length',
     'up_depth',
     'down_depth',
        ],
    continuous = [
         'descendants_relative'
        ]
)