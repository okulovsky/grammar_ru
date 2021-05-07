from ..common.nlp_algorithm import NlpAlgorithm
import pandas as pd
from ..externals import TikhonovDict, PyMorphyFeaturizer



class RepetitionsAlgorithm(NlpAlgorithm):
    def __init__(self,
                 vicinity: int = 50,
                 allow_simple_check = True,
                 allow_normal_form_check = True,
                 allow_tikhonov_check = True
                 ):
        super(RepetitionsAlgorithm, self).__init__('repetition_status', None)
        self.vicinity = vicinity
        tic = TikhonovDict.read_as_df()
        tic = tic.loc[tic.morpheme_type == 'ROOT']
        self.tic = tic.set_index('word').morpheme.to_frame('value')
        self.allow_simple_check = allow_simple_check
        self.allow_normal_form_check = allow_normal_form_check
        self.allow_tikhonov_check = allow_tikhonov_check


    def generate_merge_index(self, df):
        ldf = df.loc[df.check_requested]
        merge_index = []
        for i in ldf.word_id:
            for j in range(max(0, i - self.vicinity), i):
                merge_index.append((i, j))
        mdf = pd.DataFrame(merge_index, columns=['word_id', 'another_id'])
        norm_id = list(df.loc[df.word_type == 'ru'].word_id)
        mdf = mdf.loc[mdf.word_id.isin(norm_id)]
        mdf = mdf.loc[mdf.another_id.isin(norm_id)]
        return mdf

    def compare(self, merge_df, word_df):
        cdf = merge_df.merge(word_df, left_on='word_id', right_index=True)
        cdf = cdf.merge(word_df, left_on='another_id', right_index=True)
        cdf['match'] = cdf.value_x == cdf.value_y
        errors = cdf.groupby('word_id').match.any()
        return list(errors.loc[errors].index)

    def run(self, df):
        mdf = self.generate_merge_index(df)
        rdf = df.loc[df.word_id.isin(mdf.word_id) | df.word_id.isin(mdf.another_id)]
        errors = []

        if self.allow_simple_check:
            word_df = rdf.set_index('word_id').word.str.lower().to_frame('value')
            errors += self.compare(mdf, word_df)

        if self.allow_normal_form_check or self.allow_tikhonov_check:
            normal_df = PyMorphyFeaturizer().create_features(rdf).set_index('word_id').normal_form.to_frame('value')

            if self.allow_normal_form_check:
                errors+= self.compare(mdf, normal_df)

            if self.allow_tikhonov_check:

                tik_df =  normal_df.rename(columns={'value':'word'}).merge(self.tic, left_on='word', right_index=True).drop('word',axis=1)
                errors += self.compare(mdf, tik_df)

        df[self.get_status_column()] = True
        df.loc[df.word_id.isin(errors), self.get_status_column()] = False




