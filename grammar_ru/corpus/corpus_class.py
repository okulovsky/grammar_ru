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

class Corpus:
    def __init__(self, location: Optional[Path], id_shift = 0):
        self.location = location
        self.id_shift = 0

    def _read_frame(self, fname):
        with zipfile.ZipFile(self.location, 'r') as file:
            buffer = BytesIO(file.read(fname))
            df = pd.read_parquet(buffer)
            return df

    def get_toc(self):
        with zipfile.ZipFile(self.location, 'r') as file:
            buffer = BytesIO(file.read('toc.parquet'))
            df = pd.read_parquet(buffer)
            return df

    def _get_frames_iter(self, uids):
        with zipfile.ZipFile(self.location, 'r') as file:
            for uid in uids:
                buffer = BytesIO(file.read(f'raw/{uid}.parquet'))
                df = pd.read_parquet(buffer)
                if self.id_shift>0:
                    for column in ['word_id','sentence_id','paragraph_id']:
                        df[column]+=self.id_shift
                    df.index = list(df.word_id)
                yield  df


    def get_frames(self, uids):
        return Queryable(self._get_frames_iter(uids), len(uids))












