from .architecture import *
from pathlib import Path

class FrequencyFeaturizer(SimpleFeaturizer):
    def __init__(self):
        super(FrequencyFeaturizer, self).__init__('frequencies')
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

    def _featurize_inner(self, db: DataBundle):
        df = db.data_frames['src']
        df = df.merge(db.data_frames['pymorphy'].normal_form, left_on='word_id', right_index=True)
        df = df.merge(self.freqs, left_on='normal_form', right_index=True, how='left')
        df = df.fillna(0)
        df = df.set_index('word_id')[self.freq_columns]
        return df