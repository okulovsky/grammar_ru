from typing import *
from slovnet.model.emb import NavecEmbedding
import torch
from navec import Navec
import pandas as pd
from ....grammar_ru.common import Loc
import os
import urllib.request
from .architecture import Featurizer, DataBundle

def download_dependency(fname, url, disable_downloading):
    path = Loc.dependencies_path/fname
    if not os.path.exists(path) and not disable_downloading:
        urllib.request.urlretrieve(url, filename=path)


class GloveFeaturizer(Featurizer):
    def __init__(self,
                 disable_downloading = False,
                 add_lowercase = True,
                 add_normal_form = True
                 ):
        download_dependency('glove_featurizer_navec', 'https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar', disable_downloading)
        self.navec = Navec.load(Loc.data_cache_path / 'glove.tar')
        self.words = list(self.navec.vocab.words)
        self.ndf = pd.DataFrame(dict(word=self.words)).reset_index(drop=False).set_index('word').rename(columns={'index': 'glove_index'})
        self.add_lowercase = add_lowercase
        self.add_normal_form = add_normal_form

    def get_frame_names(self) -> List[str]:
        return ['glove_keys','glove_scores']

    def featurize(self, db: DataBundle) -> None:
        df = db.src.set_index('word_id')[['word', 'word_type']]
        check_columns = ['word']

        if self.add_lowercase:
            df['lowercase_word'] = df.word.str.lower()
            check_columns.append('lowercase_word')

        if self.add_normal_form:
            df = df.merge(db.pymorphy[['normal_form']], left_index=True, right_index=True)
            check_columns.append('normal_form')


        df = df.loc[df.word_type == 'ru'].copy()

        UNK = '<unk>'
        df['selected_word'] = UNK
        df['selected_word_column'] = UNK
        for column in check_columns:
            lc = (df.selected_word == UNK) & df[column].isin(self.words)
            df.loc[lc, 'selected_word'] = df.loc[lc][column]
            df.loc[lc, 'selected_word_column'] = column

        df = df.merge(self.ndf, left_on='selected_word', right_index=True)
        df = df[['selected_word', 'selected_word_column', 'glove_index']]
        db['glove_keys'] = df

        emb = NavecEmbedding(self.navec)
        t_input = torch.tensor(list(df.glove_index.unique()))
        t_output = emb(t_input)
        gdf = pd.DataFrame(t_output.tolist())
        gdf['glove_index'] = t_input.tolist()
        gdf = gdf.set_index('glove_index')
        gdf.columns = [f'c{c}' for c in gdf.columns]
        db['glove_scores'] = gdf
