from .architecture import NlpAlgorithm
from ..ml.features import \
    PyMorphyFeaturizer, \
    SlovnetFeaturizer, \
    NavecFeaturizer, \
    SimpleFeaturizer
import pandas as pd
import numpy as np
from ...common import DataBundle


class AntecedentCandidatesAlgorithm(NlpAlgorithm):
    def __init__(self, dbs_query=None, max_candidates: int = 10):
        self.pmf = PyMorphyFeaturizer()
        self.slvnt = SlovnetFeaturizer()
        self.navec = NavecFeaturizer()
        self.max_candidates = max_candidates
        if dbs_query is not None:
            self.pronoun_replacer = self._get_proper_replace_dict(dbs_query)

    def _get_proper_replace_dict(self, dbs_query):
        proper_counts = pd.DataFrame()
        for db in dbs_query:
            db_counts = self._count_propers(db)
            proper_counts = \
                (pd.concat([proper_counts, db_counts])
                 .groupby(['normal_form', 'gender'])
                 .sum()
                 .reset_index())
        converter = {'femn': 'женщина',
                     'masc': 'мужчина',
                     'neut': 'существо',
                     None: 'существо'}
        return dict(zip(proper_counts.normal_form,
                        proper_counts.gender.apply(lambda x: converter[x])))

    def pronoun_replace_for_db(self, db):
        converter = {'femn': 'женщина',
                     'masc': 'мужчина',
                     'neut': 'существо',
                     None: 'существо'}
        db_counts = self._count_propers(db)
        proper_counts =\
            (db_counts
             .groupby(['normal_form', 'gender'])
             .sum()
             .reset_index())
        self.pronoun_replacer = dict(zip(proper_counts.normal_form,
                                         proper_counts.gender.apply(lambda x: converter[x])))

    @staticmethod
    def _src(db: DataBundle):
        try:
            return db['speechless_src']
        except KeyError:
            return db['src']

    def _is_proper(self, noun: str, db: DataBundle):
        indices = list(db.pymorphy[db.pymorphy['normal_form'] == noun].index)
        src = self._src(db)
        words = src[src.index.isin(indices)]['word']
        return words.str[0].str.isupper().all() and words.count() > 2

    def _count_propers(self, db: DataBundle):
        self.pmf.featurize(db)
        nouns = db.pymorphy[db.pymorphy['POS'] == 'NOUN']
        counted = nouns.groupby(['normal_form'])['normal_form'].count()
        propers = counted[counted.index.map(lambda x: self._is_proper(x, db))]
        propers = propers.to_frame().rename(columns={'normal_form': 'count'})
        propers = propers.reset_index()
        proper_rows = db.pymorphy[db.pymorphy.normal_form.isin(
            propers['normal_form'])
        ][['normal_form', 'gender']].drop_duplicates()
        return propers.merge(proper_rows, on='normal_form')

    @staticmethod
    def _get_pronouns(morphology_df: pd.DataFrame):
        pronouns_filter = morphology_df['normal_form'].isin(
            ['он', 'она', 'оно'])
        pronouns_df = morphology_df[pronouns_filter][['gender']]
        pronouns_df = pronouns_df.reset_index()
        return pronouns_df.add_prefix('pronoun_')

    def _get_antecedent_candidates(self, db: DataBundle):
        slovnet = self._get_features(db, 'slovnet', self.slvnt)
        cand_filter = (slovnet['POS'].isin(['NOUN', 'PROPN']) &
                       (db.slovnet['Number'] == 'Sing'))
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
        merged_df = merged_df[
            merged_df['candidate_distance'] < self.max_candidates]
        return merged_df[['pronoun_word_id',
                          'candidate_word_id',
                          'candidate_distance']].reset_index(drop=True)

    @staticmethod
    def _get_features(db: DataBundle, frames_name: str,
                      featurizer: SimpleFeaturizer):
        try:
            return db.data_frames[frames_name]
        except KeyError:
            featurizer.featurize(db)
            return db.data_frames[frames_name]

    def get_candidates(self, db: DataBundle):
        morphology_df = self._get_features(db, 'pymorphy', self.pmf)
        pronouns_df = self._get_pronouns(morphology_df)
        antecedent_candidates_df = self._get_antecedent_candidates(db)
        return self._filter_candidates(pronouns_df, antecedent_candidates_df)

    def get_pronoun_parent(self,
                           db: DataBundle,
                           candidates_df: pd.DataFrame = None):
        if candidates_df is None:
            candidates_df = self.get_candidates(db)
        slovnet = self._get_features(db, 'slovnet', self.slvnt)

        parent_ids = slovnet[
            slovnet.index.isin(
                candidates_df['pronoun_word_id'])]['syntax_parent_id']
        parent_df = (parent_ids.to_frame()
                     .reset_index()
                     .rename(columns={'syntax_parent_id': 'pronoun_parent_id',
                                      'word_id': 'pronoun_word_id'}))
        return candidates_df.merge(parent_df, on='pronoun_word_id')

    @staticmethod
    def get_big_deviation_only(df: pd.DataFrame):
        product_groups = df.groupby(['pronoun_word_id'])['score']
        maxes = product_groups.max()
        means = product_groups.mean()
        stds = product_groups.std()
        max_indices = product_groups.idxmax().dropna().astype(int)
        best_in_group = df.loc[max_indices]
        temp = ((product_groups.count() < 3) | (
                stds < maxes - means)).to_frame()
        res = best_in_group[
            best_in_group.pronoun_word_id.isin(temp[temp['score']].index)]
        return res.reset_index(drop=True)

    def inflect(self, word, case):
        try:
            return self.pmf.an.parse(word)[0].inflect({'sing', case}).word
        except:
            return None

    def get_inflected_candidates(self, pymorphy, with_parents_df):
        inflect_base = \
            with_parents_df.merge(pymorphy, left_on='pronoun_word_id',
                                  right_on='word_id')[['candidate_word_id',
                                                       'case']]
        inflect_base = inflect_base.merge(
            pymorphy.reset_index()[['word_id', 'normal_form']],
            left_on='candidate_word_id', right_on='word_id', how='left')[
            ['case', 'normal_form']]
        return inflect_base.apply(
            lambda x: self.inflect(x['normal_form'], x['case']), axis=1).dropna()

    def remove_sibling_candidates(self, db, with_parents_df):
        slovnet = \
            self._get_features(db, 'slovnet', self.slvnt)[['syntax_parent_id']]
        cmp = (with_parents_df
               .merge(slovnet, left_on='candidate_word_id', right_index=True, how='left'))
        return with_parents_df[
            cmp['syntax_parent_id'] != cmp['pronoun_parent_id']
        ].reset_index(drop=True)

    def get_glove_prod_e_norm(self, df, col1, col2):
        copy = df.copy()
        copy['prod_e'] = self.navec.get_glove_prod(copy, col1, col2).apply(
            lambda x: np.e ** x)
        return copy.groupby('pronoun_word_id')['prod_e'].transform(
            lambda x: x / x.sum()).fillna(0)

    def get_context_words(self, db, with_parents_df, neighb_count=1):
        extended_df = with_parents_df.copy()
        src = self._src(db)

        for i in range(1, neighb_count + 1):
            left_id = 'left_id_' + str(i)
            extended_df[left_id] = \
                with_parents_df['pronoun_word_id'].apply(
                    lambda x:
                    x - i if (x - i >= 0 and
                              src.loc[x - i].word_type != 'punct') else -1)
            right_id = 'right_id_' + str(i)
            extended_df[right_id] = \
                with_parents_df['pronoun_word_id'].apply(
                    lambda x:
                    x + i if (x + i < len(src.index) and
                              src.loc[x + i].word_type != 'punct')
                    else -1)

            extended_df = extended_df.merge(
                src.word.str.lower().rename('left_word_' + str(i)),
                left_on=left_id, right_index=True, how='left').reset_index(drop=True)
            extended_df = extended_df.merge(
                src.word.str.lower().rename('right_word_' + str(i)),
                left_on=right_id, right_index=True, how='left').reset_index(
                drop=True)

        extended_df = extended_df.merge(
            src.word.str.lower().rename('parent_word'),
            left_on='pronoun_parent_id', right_index=True, how='left')
        return extended_df

    def get_result_score(self, db, with_parents_df, neighb_count=1, coeff=1):
        extended_df = self.get_context_words(db, with_parents_df, neighb_count)

        result_series = self.get_glove_prod_e_norm(extended_df,
                                                   'candidate',
                                                   'parent_word')
        for i in range(1, neighb_count + 1):
            i_addendum = (self.get_glove_prod_e_norm(extended_df,
                                                     'candidate',
                                                     'left_word_' + str(i)) +
                          self.get_glove_prod_e_norm(extended_df,
                                                     'candidate',
                                                     'right_word_' + str(i)))
            result_series += i_addendum * coeff
        return result_series

    def filter_max_first_far_from_second(self, df):
        first_max_indices = df.groupby('pronoun_word_id')['score'].idxmax()
        first_maxes = df.loc[first_max_indices][['pronoun_word_id', 'score']]
        without_maxes = df.drop(first_max_indices.to_list())
        second_maxes = without_maxes.loc[
            without_maxes.groupby('pronoun_word_id')['score'].idxmax()][
            ['pronoun_word_id', 'score']]
        cmp = first_maxes.merge(second_maxes, on='pronoun_word_id',
                                how='left').fillna(0)
        cmp['diff'] = cmp['score_x'] - cmp['score_y']
        cmp['std'] = df.groupby('pronoun_word_id')[
            'score'].std().fillna(0).reset_index(drop=True)
        good_diff = cmp[cmp['diff'] > cmp['std']]['pronoun_word_id'].to_list()
        return first_max_indices[good_diff].to_list()

    def run_full(self,
                 db: DataBundle,
                 inflect=True,
                 neighbour_count=1,
                 neighbour_coeff=1,
                 use_diff_between_second_filter=True,
                 use_diff_between_mean_filter=False):
        self.pmf.featurize(db)
        db.pymorphy['normal_form'] = db.pymorphy['normal_form'].apply(
            lambda x:
            self.pronoun_replacer[x] if x in self.pronoun_replacer else x)
        candidates_df = self.get_candidates(db)
        with_parents_df = self.get_pronoun_parent(db, candidates_df)
        with_parents_df = self.remove_sibling_candidates(db, with_parents_df)
        if inflect:
            with_parents_df['candidate'] = \
                self.get_inflected_candidates(db.pymorphy, with_parents_df)
        else:
            with_parents_df = with_parents_df.merge(
                db.pymorphy['normal_form'].rename('candidate'),
                left_on='candidate_word_id', right_index=True, how='left')
        with_parents_df = (with_parents_df
                           .drop_duplicates(['candidate', 'pronoun_word_id'],
                                            keep='last')
                           .reset_index(drop=True))

        filtered_df = with_parents_df.copy()
        filtered_df['score'] = \
            self.get_result_score(db, with_parents_df, neighbour_count, neighbour_coeff)
        if use_diff_between_second_filter:
            filtered_df = filtered_df.loc[
                self.filter_max_first_far_from_second(
                    filtered_df)].reset_index(drop=True)
        if use_diff_between_mean_filter:
            filtered_df = self.get_big_deviation_only(filtered_df)

        return candidates_df, filtered_df

    def _run_inner(self, db: DataBundle, index: pd.Index):
        pass
