import numpy as np
import pandas as pd

from tg.common import DataBundle
from tg.common.ml.batched_training import train_display_test_split
from tg.grammar_ru.ml.features import PyMorphyFeaturizer
import pymorphy2


class GGTrainIndexBuilder:

    def __init__(self):
        self.pmf = PyMorphyFeaturizer()
        self.an = pymorphy2.MorphAnalyzer()  # TODO delete
        self.speech_part_labels = ['NOUN', 'ADJF', 'ADJS', 'VERB', 'PRTF', 'PRTS']  # TODO what else?
        self.gender_nums = {g: i for i, g in enumerate(['masc', 'femn', 'neut', 'nan'])}

    def build_train_index(self, df):
        df['label'] = -1
        db = DataBundle(src=df)
        self.pmf.featurize(db)# TODO GenderLabelPyMorphyFeaturizer
        morphed = db.data_frames['pymorphy']
        morphed.replace({np.nan: 'nan'}, inplace=True)
        df['is_target'] = morphed.POS.isin(self.speech_part_labels)
        df.loc[df.is_target, 'label'] = morphed[df.is_target].gender.replace(self.gender_nums)
        return [df]

    @staticmethod
    def build_index_from_src(src_df):
        df = src_df.loc[src_df.is_target][['word_id', 'sentence_id', 'label']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
