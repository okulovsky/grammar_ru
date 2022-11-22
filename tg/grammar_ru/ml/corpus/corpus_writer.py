from typing import *
import pandas as pd
from pathlib import Path
import zipfile
from datetime import datetime
from io import BytesIO
import os
from uuid import uuid4
from ...common import DataBundle, Separator
from yo_fluq_ds import Query
import time

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
    def __init__(self,
                 filename: Path,
                 overwrite = False,
                 recompute_ids_with_span: Optional[int] = 10000,
                 ):
        if filename.is_file():
            if not overwrite:
                raise ValueError(f'{filename} exists')
            else:
                os.remove(filename)
        os.makedirs(filename.parent, exist_ok=True)
        self.file = zipfile.ZipFile(filename,'w',zipfile.ZIP_DEFLATED)
        self.toc = []
        self.indices = {}
        self.ordinal = 0
        self.recompute_ids_with_span = recompute_ids_with_span

    def _write_parquet(self, name, df: pd.DataFrame):
        bytes = BytesIO()
        df.to_parquet(bytes)
        self.file.writestr(name, bytes.getbuffer())

    def _update_indices(self, df):
        if self.recompute_ids_with_span is None:
            return df
        if len(self.toc)>0:
            delta = self.toc[-1]['max_id'] + self.recompute_ids_with_span
            df = Separator.reset_indices(df, delta)
        return df



    def add_fragment(self, fragment: Union[CorpusFragment,pd.DataFrame]):
        if isinstance(fragment, pd.DataFrame):
            fragment = CorpusFragment('', 0, fragment, {})

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

        fragment.df = self._update_indices(fragment.df)
        row['max_id'] = Separator.get_max_id(fragment.df)

        Separator.validate(fragment.df)
        self._write_parquet(f'src/{file_id}.parquet', fragment.df)
        self.toc.append(row)




    def finalize(self, custom_toc=None):
        if custom_toc is None:
            toc = pd.DataFrame(self.toc)
            toc = toc.set_index('file_id')
        else:
            toc = custom_toc
        self._write_parquet('toc.parquet',toc)
        self.file.close()


    @staticmethod
    def collect_from_files(folder, target_file):
        folder = Path(folder)
        with zipfile.ZipFile(target_file, 'w', zipfile.ZIP_DEFLATED) as zp:
            for in_file_name in Query.folder(folder,'**/*'):
                if not in_file_name.is_file():
                    continue
                with open(in_file_name,'rb') as in_file:
                    bytes = in_file.read()
                    relative_path = in_file_name.relative_to(folder)
                    zp.writestr(str(relative_path), bytes)



