from .architecture import NlpAlgorithm
from ..ml.features import \
    PyMorphyFeaturizer, \
    SlovnetFeaturizer, \
    NavecFeaturizer, \
    SimpleFeaturizer
import pandas as pd
from ...common import DataBundle


class AntecedentCandidatesAlgorithm(NlpAlgorithm):
    def __init__(self, dbs_query, max_candidates: int = 10):
        self.pmf = PyMorphyFeaturizer()
        self.slvnt = SlovnetFeaturizer()
        self.navec = NavecFeaturizer()
        self.max_candidates = max_candidates
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

    @staticmethod
    def _is_proper(noun: str, db: DataBundle):
        indices = list(db.pymorphy[db.pymorphy['normal_form'] == noun].index)
        words = db.src[db.src.index.isin(indices)]['word']
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
        merged_df = merged_df[merged_df['candidate_distance'] < self.max_candidates]
        return merged_df[['pronoun_word_id',
                          'candidate_word_id',
                          'candidate_distance']].reset_index(drop=True)

    @staticmethod
    def _get_features(db: DataBundle, frames_name: str, featurizer: SimpleFeaturizer):
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
    def get_big_deviation_only(df: pd.DataFrame, col1: str, col2: str, prod_col: str):
        product_groups = \
        df.drop_duplicates([col1, col2],
                           keep='last').groupby(['pronoun_word_id'])[prod_col]
        maxes = product_groups.max()
        means = product_groups.mean()
        stds = product_groups.std()
        max_indices = product_groups.idxmax().dropna().astype(int)
        best_in_group = df.loc[max_indices]
        temp = ((product_groups.count() < 4) | (
                    stds < maxes - means)).to_frame()
        return best_in_group[
            best_in_group.pronoun_word_id.isin(temp[temp['product']].index)]

    def inflect(self, word, case):
        return self.pmf.an.parse(word)[0].inflect({'sing', case}).word

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
            lambda x: self.inflect(x['normal_form'], x['case']), axis=1)

    def filter_glove(self, with_parents_df: pd.DataFrame, db: DataBundle):
        product_df = with_parents_df.merge(
            db.src.word.str.lower().rename('pronoun_parent'),
            left_on='pronoun_parent_id', right_index=True, how='left')
        product_df['inflected_candidate'] = \
            self.get_inflected_candidates(db.pymorphy, with_parents_df)
        featurizer = NavecFeaturizer()
        product_df['product'] = featurizer.get_glove_prod(product_df,
                                                          'pronoun_parent',
                                                          'inflected_candidate')
        a = self.get_big_deviation_only(
            product_df, 'pronoun_parent', 'inflected_candidate', 'product')
        return self.get_big_deviation_only(
            product_df, 'pronoun_parent', 'inflected_candidate', 'product')

    def filter_glove_neighbour(self, candidates_df: pd.DataFrame, db: DataBundle):
        product_df = candidates_df.merge(
            db.src.word.str.lower().rename('pronoun_parent'),
            left_on='pronoun_parent_id', right_index=True, how='left')

    def run_full(self, db: DataBundle):
        self.pmf.featurize(db)
        db.pymorphy['normal_form'] = db.pymorphy['normal_form'].apply(
            lambda x:
            self.pronoun_replacer[x] if x in self.pronoun_replacer else x)
        candidates_df = self.get_candidates(db)
        with_parents_df = self.get_pronoun_parent(db, candidates_df)
        return self.filter_glove(with_parents_df, db)

    def _run_inner(self, db: DataBundle, index: pd.Index):
        pass
