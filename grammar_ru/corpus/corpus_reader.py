from typing import *
from pathlib import Path
import os
import zipfile
from uuid import uuid4
import json
import pandas as pd
from io import BytesIO
from yo_fluq_ds import *
import pickle
from tg.common.ml import batched_training as bt

class CorpusReader:
    def __init__(self, location: Optional[Path], id_shift = 0):
        self.location = location
        self.id_shift = id_shift
        self.updatable_columns = ['word_id', 'sentence_id', 'paragraph_id']

    def _read_frame(self, file, fname):
        buffer = BytesIO(file.read(fname))
        df = pd.read_parquet(buffer)
        return df


    def get_toc(self):
        with zipfile.ZipFile(self.location, 'r') as file:
            return self._read_frame(file,'toc.parquet')


    def _read_src(self, file, uid):
        df = self._read_frame(file, f'src/{uid}.parquet')
        if self.id_shift > 0:
            for column in self.updatable_columns:
                df[column] += self.id_shift
            df.index = list(df.word_id)
        df['file_id'] = uid
        return df

    def _get_frames_iter(self, uids):
        with zipfile.ZipFile(self.location, 'r') as file:
            for uid in uids:
                yield  self._read_src(file, uid)


    def get_frames(self, uids):
        return Queryable(self._get_frames_iter(uids), len(uids))


    def _get_bundles_iter(self, uids):
        with zipfile.ZipFile(self.location, 'r') as file:
            buffer = BytesIO(file.read('featurizers.json'))
            featurizers = json.load(buffer)
            for uid in uids:
                frames = {}
                frames['src'] = self._read_src(file, uid)
                for featurizer in featurizers:
                    df = self._read_frame(file, f'{featurizer}/{uid}.parquet')
                    if df.index.name not in self.updatable_columns:
                        raise ValueError(f'Featurizer {featurizer} has produced a DF with the index {df.index.name} which is not updatable')
                    if self.id_shift is not None:
                        df.index = df.index+self.id_shift
                    frames[featurizer] = df
                bundle = bt.DataBundle(None,frames)
                yield bundle


    def get_bundles(self, uids):
        return Queryable(self._get_bundles_iter(uids), len(uids))













