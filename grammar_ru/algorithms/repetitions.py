from .architecture import NlpAlgorithm
from ..ml.features import PyMorphyFeaturizer, MorphemeTikhonovEnricher
from ..common import DataBundle
from yo_fluq_ds import *



class RepetitionsAlgorithm(NlpAlgorithm):
    def __init__(self,
                 vicinity: int = 50,
                 allow_simple_check = True,
                 allow_normal_form_check = True,
                 allow_tikhonov_check = True,
                 ):
        super(RepetitionsAlgorithm, self).__init__('repetition_status', None, 'repetition_hint')
        self.vicinity = vicinity
        self.allow_simple_check = allow_simple_check
        self.allow_normal_form_check = allow_normal_form_check
        self.allow_tikhonov_check = allow_tikhonov_check
        if self.allow_normal_form_check or self.allow_tikhonov_check:
            self.pymorphy = PyMorphyFeaturizer().as_enricher()
        if self.allow_tikhonov_check:
            self.tikhonov = MorphemeTikhonovEnricher(['ROOT'])


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
        cdf = cdf.loc[cdf.value_x == cdf.value_y]
        if cdf.shape[0]==0:
            return None
        cdf = cdf.feed(fluq.add_ordering_column('word_id', ('another_id', False), 'order'))
        cdf = cdf.loc[cdf.order == 0]
        cdf = cdf[['word_id', 'another_id']]
        return cdf


    def _add_algorithm(self, df, merge_df, word_df, algorithm_name):
        cdf = self.compare(merge_df, word_df)
        if cdf is None:
            return
        condition = df[self.get_status_column()] & df.word_id.isin(cdf.word_id)
        df.loc[condition,self.get_status_column()] = False

        df.loc[condition,'repetition_algorithm'] = algorithm_name
        ids = list(df.loc[condition].word_id)
        ref = cdf.set_index('word_id').another_id
        for id in ids:
            df.loc[df.word_id==id,'repetition_reference'] = ref[id]

    def run_on_bundle(self, db: DataBundle):
        df = db.src
        mdf = self.generate_merge_index(df)
        rdf = df.loc[df.word_id.isin(mdf.word_id) | df.word_id.isin(mdf.another_id)]

        df[self.get_status_column()] = True

        df['repetition_reference'] = -1
        df['repetition_algorithm'] = None

        if self.allow_simple_check:
            word_df = rdf.set_index('word_id').word.str.lower().to_frame('value')
            self._add_algorithm(df, mdf, word_df, 'simple')

        if self.allow_normal_form_check or self.allow_tikhonov_check:
            if self.pymorphy.get_df_name() not in db:
                self.pymorphy.enrich(db)

            if self.allow_normal_form_check:
                normal_df = db.pymorphy.normal_form.to_frame('value')
                self._add_algorithm(df, mdf, normal_df, 'normal')

            if self.allow_tikhonov_check:
                self.tikhonov.enrich(db)
                tikhonov_df = db[self.tikhonov.get_df_name()].morpheme.to_frame('value')
                self._add_algorithm(df, mdf, tikhonov_df, 'tikhonov')

        err = ~df.repetition_status
        df['repetition_hint'] = ''
        ref_df = df.loc[err].repetition_reference.to_frame().merge(df.set_index('word_id').word, left_on='repetition_reference', right_index=True).word

        df.loc[err, 'repetition_hint'] = (
                'On algorithm `'+
                df.loc[err,'repetition_algorithm']+
                '` collides with word `'+
                ref_df+
                '` located '+
                (df.loc[err,'word_id'] - df.loc[err,'repetition_reference']).astype(str)

        )


    def _run_inner(self, df):
        self.run_on_bundle(DataBundle(src=df))

