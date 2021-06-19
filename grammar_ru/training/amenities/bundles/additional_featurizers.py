import pandas as pd
from yo_fluq_ds import *
from pathlib import Path

class WordFrequencyFeaturizer:
    def __init__(self):
        path = Path(__file__).parent/'frequency_dict.csv'
        fdf = pd.read_csv(path, sep='\t')
        fdf.columns = ['lemma', 'pos', 'freq', 'corpus_range', 'juilland_coef', 'text_count']
        fdf.lemma = fdf.lemma.str.lower()
        fdf = fdf.groupby('lemma').aggregate(
            dict(freq='sum', corpus_range='max', juilland_coef='max', text_count='max'))
        fdf = fdf.sort_values('freq', ascending=False)
        fdf['freq_found'] = 1
        self.freq_columns = list(fdf.columns)
        self.freqs = fdf

    def __call__(self, db):
        df = db.data_frames['src']
        df = df.merge(db.data_frames['pymorphy'].normal_form, left_on='word_id', right_index=True)
        df = df.merge(self.freqs, left_on='normal_form', right_index=True, how='left')
        df = df.fillna(0)
        df = df.set_index('word_id')[self.freq_columns]
        db.data_frames['frequencies'] = df


def add_local_freq(db):
    src = db.data_frames['src']
    pym = db.data_frames['pymorphy']

    df = src[['word_id']].merge(pym[['normal_form']],left_on='word_id', right_index=True)
    freq = (df.groupby('normal_form').size()/df.shape[0]).to_frame('freq')
    freq['freq_found'] = 1
    df = df.merge(freq, left_on='normal_form', right_index=True, how='left')
    df = df.set_index('word_id').drop('normal_form',axis=1).fillna(0)
    db.data_frames['local_freq'] = df


def add_capitalization_data(db):
    df = db.data_frames['src']
    df['is_capitalized'] = df.word.str.slice(0,1).str.lower()!=df.word.str.slice(0,1)
    tdf = df.loc[df.word_type=='ru']
    tdf = tdf.feed(fluq.add_ordering_column('sentence_id','word_index','ru_word_order'))
    df = df.merge(tdf.ru_word_order, left_index=True, right_index=True, how='left')
    df.ru_word_order = df.ru_word_order.fillna(-1).astype(int)
    df['is_unexpectedly_capitalized'] = df.is_capitalized & (df.ru_word_order!=0)
    df = df.merge(db.data_frames['pymorphy'].normal_form, left_on='word_id',right_index=True)
    df = df.merge(
        df.groupby('normal_form').is_unexpectedly_capitalized.any().to_frame('uc_somewhere'),
        left_on='normal_form',
        right_index=True)
    df = df.merge(
        df.groupby('normal_form').is_unexpectedly_capitalized.mean().to_frame('uc_proportion'),
        left_on='normal_form',
        right_index=True)
    df = df.set_index('word_id')[['is_capitalized','is_unexpectedly_capitalized', 'uc_somewhere','uc_proportion']]
    db.data_frames['capitalization'] = df

