from yo_fluq_ds import *

class ControlDistribution:
    def __init__(self, src, column, reference_df=None):
        self.src = src
        self.column = column
        self.reference_df = reference_df

    def get_row(self, db):
        data = db.data_frames[self.src].groupby(self.column).feed(fluq.fractions())
        data = dict(data)
        data['file_id'] = db.data_frames['src'].file_id.iloc[0]
        return data

    def build_stat_table(self, bundles):
        data = [self.get_row(db) for db in bundles]
        df = pd.DataFrame(data).set_index('file_id').fillna(0)
        df.columns = [self.column + '_' + c for c in df.columns]
        return df

    @staticmethod
    def build_reference_table(df):
        sdf = (df
               .aggregate(['mean', 'std'])
               .transpose()
               .reset_index()
               .rename(columns=dict(index='parameter', mean='mu', std='sigma'))
               .set_index('parameter')
               .sort_values('mu')
               )
        sdf['zero'] = 0
        return sdf

    def train(self, df):
        self.reference_df = ControlDistribution.build_reference_table(df)

    def build_deviation_table(self, df):
        ddf = (df + self.reference_df.zero).fillna(0)
        ndf = ((ddf - self.reference_df.mu) / self.reference_df.sigma).abs()
        ndf.columns = ['deviation_' + c for c in ndf.columns]
        cols = ndf.columns
        ddf = ddf.merge(ndf, left_index=True, right_index=True)
        ddf[f'max_{self.column}_deviation'] = ddf[cols].max(axis=1)
        ddf[f'sum_{self.column}_deviation'] = ddf[cols].sum(axis=1)
        return ddf
