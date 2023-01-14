from .architecture import  *
import pandas as pd
from yo_fluq_ds import fluq

class CapitalizationFeaturizer(SimpleFeaturizer):
    def __init__(self):
        super(CapitalizationFeaturizer, self).__init__('capitalization', False)

    def _featurize_inner(self, db: DataBundle):
        df = db.data_frames['src']
        df['is_capitalized'] = df.word.str.slice(0,1).str.lower()!=df.word.str.slice(0,1)
        tdf = df.loc[df.word_type=='ru']
        tdf = tdf.feed(fluq.add_ordering_column('sentence_id','word_index','ru_word_order'))
        df = df.merge(tdf.ru_word_order, left_index=True, right_index=True, how='left')
        df.ru_word_order = df.ru_word_order.fillna(-1).astype(int)
        df['is_unexpectedly_capitalized'] = df.is_capitalized & (df.ru_word_order!=0)
        df = df.merge(db.data_frames['pymorphy'].normal_form, left_on='word_id',right_index=True)
        df = df.merge(
            df.groupby('normal_form').is_unexpectedly_capitalized.mean().to_frame('uc_proportion'),
            left_on='normal_form',
            right_index=True)
        df = df.set_index('word_id')[['is_capitalized','is_unexpectedly_capitalized', 'uc_proportion']]
        return df
