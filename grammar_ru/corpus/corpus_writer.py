from typing import *
import pandas as pd
from pathlib import Path
import zipfile
from datetime import datetime
from io import BytesIO
import os
from uuid import uuid4

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

        self.shifts = dict(word_id=0, sentence_id=0, paragraph_id=0)
        self.id_span = id_span



    def _write_parquet(self, name, df: pd.DataFrame):
        bytes = BytesIO()
        df.to_parquet(bytes)
        self.file.writestr(name, bytes.getbuffer())

    def _update_indices(self, df):
        for key in list(self.shifts):
            df[key]+=self.shifts[key]
            self.shifts[key] = df[key].max()+self.id_span
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

        for key, value in fragment.additional_columns.items():
            row[key] = value
        self.toc.append(row)

        self._update_indices(fragment.df)
        self._write_parquet(f'raw/{file_id}.parquet', fragment.df)


    def finalize(self):
        toc = pd.DataFrame(self.toc)
        toc = toc.set_index('file_id')
        self._write_parquet('toc.parquet',toc)
        self.file.close()
