import numpy as np

from tg.common import DataBundle
from tg.common.ml.batched_training import train_display_test_split
from research.grammatical_gender.gender_label_pymorphy_featurizer import GenderLabelPyMorphyFeaturizer
import pymorphy2


class GGTrainIndexBuilder:

    def __init__(self):
        self.pmf = GenderLabelPyMorphyFeaturizer()  # PyMorphyFeaturizer()
        self.an = pymorphy2.MorphAnalyzer()  # TODO delete
        self.speech_part_labels = [
            'ADJF', 'ADJS', 'VERB', 'PRTF', 'PRTS']
        self.gender_nums = {g: i for i, g in enumerate(
            ['masc', 'femn', 'neut'])}  # , 'nan'])}

    def build_train_index(self, source, df, toc_row):
        df['label'] = -1
        db = DataBundle(src=df)
        self.pmf.featurize(db)  # запишет результат по ключу pymorphy
        morphed = db.data_frames['pymorphy']
        morphed.replace({np.nan: 'nan'}, inplace=True)
        a = morphed.POS.isin(self.speech_part_labels)
        b = (~df.word.str[0].str.isupper())
        g = morphed.gender
        c = (g != 'nan')
        df['is_target'] = a & b & c
        gender_scores_cols = [
            'gender_masc_score', 'gender_femn_score', 'gender_neut_score', 'gender_None_score']
        #'gender_masc_score', 'gender_femn_score', 'gender_neut_score', 'gender_None_score'
        df[gender_scores_cols] = morphed[gender_scores_cols]
        df.loc[df.is_target, 'label'] = morphed[df.is_target].gender.replace(
            self.gender_nums)
        return [df]

    @staticmethod
    def build_index_from_src(src_df):
        df = src_df.loc[src_df.is_target][[
            'word_id', 'sentence_id', 'label']].copy()
        df = df.reset_index(drop=True)
        df.index.name = 'sample_id'
        df['split'] = train_display_test_split(df)
        return df
