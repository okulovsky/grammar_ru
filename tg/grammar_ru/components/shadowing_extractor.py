from tg.common.ml import dft
from tg.common.ml import batched_training as bt


class ShadowingTransformer:
    def __init__(self,
                 inner_transformer,
                 columns_to_shadow,
                 exclusion_column=None,
                 exclusion_values=None
                 ):
        self.inner_transformer = inner_transformer
        self.columns_to_shadow = columns_to_shadow
        self.exclusion_column = exclusion_column
        self.exclusion_values = exclusion_values

    def _shadow(self, df):
        df = df.copy()
        if self.exclusion_column is not None and self.exclusion_values is not None:
            idx = ~df[self.exclusion_column].isin(self.exclusion_values)
        else:
            idx = df.index
        df.loc[idx, self.columns_to_shadow] = None
        return df

    def fit(self, df):
        df = self._shadow(df)
        self.inner_transformer.fit(df)

    def transform(self, df):
        df = self._shadow(df)
        return self.inner_transformer.transform(df)
