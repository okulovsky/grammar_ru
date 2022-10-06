import pandas as pd
import numpy as np
from yo_fluq_ds import *
from .architecture import NlpAlgorithm
from ..ml.features import PyMorphyFeaturizer, SlovnetFeaturizer, AntecedentFeaturizer
from ...common import DataBundle


class AntecedentCandidatesAlgorithm(NlpAlgorithm):
    def __init__(self, left_vicinity: int = 50):
        self.left_vicinity = left_vicinity

    def _get_pronoun_filter(self):
        return lambda x: ((x.normal_form == 'он') |
                          (x.normal_form == 'она') |
                          (x.normal_form == 'оно'))

    def _get_row_filter(self, pronoun_row: pd.Series):
        return lambda x: ((x.gender == pronoun_row.gender) &
                          (x.number == 'sing') &
                          ((x.POS == 'NOUN') |
                           (x.POS == 'ADJF') |
                           (x.POS == 'ADJS') |
                           (x.POS == 'NPRO') |
                           (x.POS == 'PRCL') |
                           (x.POS == 'PRTF') |
                           (x.POS == 'PRTS') |
                           (x.POS == 'ADVB')))

    def _get_pairs_dull(self, df: pd.DataFrame):
        return df.loc[lambda x: x.sentence_distance < 2, ['word_id', 'pronoun_id']]

    def _get_anaphor_antecedent_pairs(self, df: pd.DataFrame):
        pass

    def _get_pronoun_antecedent_candidates(self, df: pd.DataFrame, row):
        start_index = max(0, row.word_id - self.left_vicinity)
        preword_df = df[(df['word_id'] >= start_index) &
                        (df['word_id'] < row.word_id)]
        candidates = preword_df.loc[self._get_row_filter(row),
                                    ['word_id', 'POS', 'animacy']]
        candidates['pronoun_id'] = row.word_id
        return candidates

    def _run_inner(self, db: DataBundle, index: pd.Index):
        pmf = PyMorphyFeaturizer()
        pmf.featurize(db)
        morphology_df = db.data_frames['pymorphy']
        pronouns_df = morphology_df.loc[self._get_pronoun_filter(), ['gender']]
        pronouns_df['word_id'] = pronouns_df.index
        word_ids = db.data_frames['src'][db.data_frames['src']['word_type'] != 'punct'][['word_id']]
        merged_df = pd.merge(word_ids,
                             morphology_df[['POS', 'animacy', 'gender', 'number']],
                             on='word_id')
        antecedent_frames = []
        for pronoun_row in Query.df(pronouns_df):
            candidates = \
                self._get_pronoun_antecedent_candidates(merged_df, pronoun_row)
            antecedent_frames.append(candidates)
        db.data_frames['candidates'] = pd.concat(antecedent_frames)
        antc = AntecedentFeaturizer()
        antc.featurize(db)
        pairs = self._get_pairs_dull(db.data_frames['antecedents'])
        antecedent_counts = pairs['pronoun_id'].value_counts()
        db.data_frames['candidates'][NlpAlgorithm.Error] = \
            db.data_frames['candidates']['pronoun_id'].map(lambda x: antecedent_counts[x] > 1)
        return db.data_frames['candidates']
