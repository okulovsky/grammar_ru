import pandas as pd
from .architecture import *
from yo_fluq_ds import *


class AntecedentFeaturizer(SimpleFeaturizer):
    def __init__(self):
        super(AntecedentFeaturizer, self).__init__('antecedents')

    def _count_distances(
            self,
            index_df: pd.DataFrame,
            antecedent_df: pd.DataFrame):
        candidate_dist = []
        word_dist = []
        sentence_dist = []
        paragraph_dist = []
        for row in Query.df(antecedent_df):
            candidate_dist.append(
                len(antecedent_df[
                        (antecedent_df['word_id'] > row.word_id) &
                        (antecedent_df['pronoun_id'] == row.pronoun_id)]))
            word_dist.append(index_df.loc[row.word_id:row.pronoun_id - 1]
                             [index_df['word_type'] != 'punct'].shape[0])
            sentence_dist.append(index_df.iloc[row.pronoun_id]['sentence_id'] -
                                 index_df.iloc[row.word_id]['sentence_id'])
            paragraph_dist.append(index_df.iloc[row.pronoun_id]['paragraph_id']
                                  - index_df.iloc[row.word_id]['paragraph_id'])
        antecedent_df['candidates_distance'] = candidate_dist
        antecedent_df['word_distance'] = word_dist
        antecedent_df['sentence_distance'] = sentence_dist
        antecedent_df['paragraph_distance'] = paragraph_dist

    def _check_actors(self,
                      index_df: pd.DataFrame,
                      antecedent_df: pd.DataFrame):
        pass


    def _featurize_inner(self, db: DataBundle):
        result_df = db.data_frames['candidates'].copy()
        self._count_distances(db.data_frames['src'], result_df)
        return result_df.reset_index(drop=True)
