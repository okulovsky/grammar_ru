from tg.grammar_ru.common import Loc
from tg.common.ml.miscellaneous.glove import GloveProcessor
from slovnet.model.emb import NavecEmbedding
from navec import Navec
import pandas as pd
import torch

class NavecFeaturizer:
    def __init__(self):
        navec = Navec.load(Loc.data_cache_path / 'glove.tar')
        words = list(navec.vocab.words)
        self.ndf = pd.DataFrame(dict(word=words)).reset_index().set_index(
            'word').rename(columns={'index': 'navec_index'})
        self.emb = NavecEmbedding(navec)

    def _set_navec_index(self, df: pd.DataFrame, df_column, index_name):
        df = df.merge(self.ndf, left_on=df_column, right_index=True, how='left')
        df.navec_index = df.navec_index.fillna(-1).astype(int)
        df = df.rename(columns={'navec_index': index_name})
        return df

    def _prepare_gdf(self, series_1, series_2):
        indices = pd.concat([series_1, series_2]).unique()
        input = torch.tensor([x for x in indices if x != -1])
        output = self.emb(input.long())
        gdf = pd.DataFrame(output.tolist())
        gdf['navec_index'] = input.tolist()
        return gdf.set_index('navec_index')

    @staticmethod
    def get_embedding(gdf, series):
        return series\
            .to_frame('idx')\
            .merge(gdf, left_on='idx', right_index=True, how='left')\
            .drop('idx', axis=1)

    def get_glove_prod(self, df, df_column_1, df_column_2):
        copied_df = df.copy()
        product_df = self._set_navec_index(copied_df, df_column_1, 'first_c')
        product_df = self._set_navec_index(product_df, df_column_2, 'second_c')
        gdf = self._prepare_gdf(product_df.first_c, product_df.second_c)
        return GloveProcessor.apply_scores(product_df.first_c, product_df.second_c, gdf)
