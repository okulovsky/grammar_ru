from typing import *
import pandas as pd
from pathlib import Path
import zipfile
from datetime import datetime
from io import BytesIO
import os
from uuid import uuid4
from ...common import DataBundle

class CorpusFragment:
    def __init__(self,
                 filename: str,
                 part_index: int,
                 df: pd.DataFrame,
                 additional_columns: Dict[str,str]
                 ):
        self.filename = filename
        self.df = df
        self.additional_columns = additional_columns


class CorpusWriter:
    def __init__(self, filename: Path, overwrite = False, id_span = 10000):
        if filename.is_file():
            if not overwrite:
                raise ValueError(f'{filename} exists')
            else:
                os.remove(filename)

        self.file = zipfile.ZipFile(filename,'w',zipfile.ZIP_DEFLATED)
        self.toc = []
        self.indices = {}
        self.ordinal = 0

        self.columns_to_shift=['word_id','sentence_id','paragraph_id']
        self.id_span = id_span




    def _write_parquet(self, name, df: pd.DataFrame):
        bytes = BytesIO()
        df.to_parquet(bytes)
        self.file.writestr(name, bytes.getbuffer())

    def _update_indices(self, df):
        delta = 0
        if len(self.toc)>0:
            delta = self.toc[-1]['max_id'] + self.id_span
        for key in self.columns_to_shift:
            df[key]+=delta
        df.index = list(df['word_id'])

    def add_fragment(self, fragment: CorpusFragment):
        if fragment.filename not in self.indices:
            self.indices[fragment.filename] = 0
        else:
            self.indices[fragment.filename]+=1
        file_id = str(uuid4())
        row = {}
        row['filename'] = str(fragment.filename)
        row['timestamp'] = datetime.now()
        row['part_index'] = self.indices[fragment.filename]
        row['file_id'] = file_id
        row['token_count'] = fragment.df.shape[0]
        row['character_count'] = fragment.df.word_length.sum()
        row['ordinal'] = self.ordinal
        self.ordinal += 1

        for key, value in fragment.additional_columns.items():
            row[key] = value

        self._update_indices(fragment.df)
        row['max_id'] = fragment.df[['word_id', 'sentence_id', 'paragraph_id']].max().max()

        self._write_parquet(f'src/{file_id}.parquet', fragment.df)
        self.toc.append(row)

    def add_bundle(self, bundle: DataBundle):
        file_id = bundle.data_frames['src'].file_id.iloc[0]
        for key, value in bundle.data_frames.items():
            self._write_parquet(f"{key}/{file_id}.parquet", value)


    def finalize(self, custom_toc=None):
        if custom_toc is None:
            toc = pd.DataFrame(self.toc)
            toc = toc.set_index('file_id')
        else:
            toc = custom_toc
        self._write_parquet('toc.parquet',toc)
        self.file.close()




