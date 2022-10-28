from yo_fluq_ds import *
from .architecture import NlpAlgorithm
from ..ml.features import PyMorphyFeaturizer, SlovnetFeaturizer
from ...common import DataBundle


class AntecedentCandidatesAlgorithm(NlpAlgorithm):
    def __init__(self, max_candidates: int = 10):
        self.max_candidates = max_candidates

    @staticmethod
    def _get_pronouns(morphology_df: pd.DataFrame):
        pronouns_filter = morphology_df['normal_form'].isin(
            ['он', 'она', 'оно'])
        pronouns_df = morphology_df[pronouns_filter][['gender']]
        pronouns_df = pronouns_df.reset_index()
        return pronouns_df.add_prefix('pronoun_')

    # @staticmethod
    # def _get_antecedent_candidates(morphology_df: pd.DataFrame):
    #     antecedent_candidate_filter = morphology_df['POS'].isin(
    #         ['NOUN', 'PRON', 'ADJF', 'ADJS', 'NPRO', 'PRCL', 'PRTF', 'PRTS',
    #          'ADVB']) & (morphology_df['number'] == 'sing')
    #     antecedent_candidates_df = morphology_df[antecedent_candidate_filter][
    #         ['gender']]
    #     antecedent_candidates_df = antecedent_candidates_df.reset_index()
    #     return antecedent_candidates_df.add_prefix('candidate_')

    @staticmethod
    def _get_antecedent_candidates(db: DataBundle):
        cand_filter = db.slovnet['POS'].isin(
            ['NOUN', 'PROPN', 'ADJ', 'PRON', 'DET']) & (db.slovnet['Number'] == 'Sing')
        candidates_df = db.pymorphy[cand_filter][['gender']]
        return candidates_df[['gender']].reset_index().add_prefix('candidate_')

    def _filter_candidates(self,
                           pronouns_df: pd.DataFrame,
                           candidates_df: pd.DataFrame):
        merged_df = pronouns_df.merge(candidates_df, how='cross')
        merged_df = merged_df[
            (merged_df['pronoun_word_id'] > merged_df['candidate_word_id']) &
            (merged_df['pronoun_gender'] == merged_df['candidate_gender'])]
        merged_df = merged_df.drop(
            columns=['pronoun_gender', 'candidate_gender']).reset_index(
            drop=True)
        merged_df['candidate_distance'] = merged_df.groupby(
            ['pronoun_word_id']).cumcount(ascending=False)
        merged_df = merged_df[merged_df['candidate_distance'] < self.max_candidates]
        return merged_df[['pronoun_word_id',
                          'candidate_word_id',
                          'candidate_distance']].reset_index(drop=True)

    def get_candidates(self, db: DataBundle):
        pmf = PyMorphyFeaturizer()
        pmf.featurize(db)
        slvnt = SlovnetFeaturizer()
        slvnt.featurize(db)
        morphology_df = db.data_frames['pymorphy']
        pronouns_df = self._get_pronouns(morphology_df)
        antecedent_candidates_df = self._get_antecedent_candidates(db)
        return self._filter_candidates(pronouns_df, antecedent_candidates_df)

    def get_pronoun_parent(self,
                           db: DataBundle,
                           candidates_df: pd.DataFrame = None):
        if candidates_df is None:
            candidates_df = self.get_candidates(db)
        slovnet = db['slovnet']

        parent_ids = slovnet[
            slovnet.index.isin(
                candidates_df['pronoun_word_id'])]['syntax_parent_id']
        parent_df = (parent_ids.to_frame()
                     .reset_index()
                     .rename(columns={'syntax_parent_id': 'pronoun_parent_id',
                                      'word_id': 'pronoun_word_id'}))
        return candidates_df.merge(parent_df, on='pronoun_word_id')

    def _run_inner(self, db: DataBundle, index: pd.Index):
        pass
