from os.path import isfile
from typing import *
import pandas as pd
from pathlib import Path
import zipfile
from datetime import datetime
from io import BytesIO
import os
from uuid import uuid4
from ..common import DataBundle, Separator
from yo_fluq_ds import Query, FileIO
from .corpus_reader import CorpusReader
import time


class CorpusFragment:
    def __init__(self,
                 filename: str,
                 part_index: int,
                 df: pd.DataFrame,
                 additional_columns: Dict[str, str]
                 ):
        self.filename = filename
        self.df = df
        self.additional_columns = additional_columns


class CorpusWriter:
    def __init__(self,
                 filename: Path,
                 overwrite=False,
                 append=False,
                 recompute_ids_with_span: Optional[int] = 10000,
                 ):
        mode = 'a' if append else 'w'
        if filename.is_file():
            if overwrite:
                os.remove(filename)
            elif not append:
                raise ValueError(f'{filename} exists')
        os.makedirs(filename.parent, exist_ok=True)
        self.filename = filename
        self.file = zipfile.ZipFile(filename, mode, zipfile.ZIP_DEFLATED)
        have_toc = 'toc.parquet' in self.file.namelist()
        if append and have_toc:
            reader = CorpusReader(filename)
            self.toc = reader.get_toc().reset_index().to_dict('records')
        else:
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
        if len(self.toc) > 0:
            delta = self.toc[-1]['max_id'] + self.recompute_ids_with_span
            df = Separator.reset_indices(df, delta)
        return df

    def _replace_toc(self,new_toc):
        zin = self.file
        zout = zipfile.ZipFile('temp.zip', 'w')
        for item in zin.infolist():
            buffer = zin.read(item.filename)
            if (item.filename != 'toc.parquet'):
                zout.writestr(item, buffer)
        zin.close()
        self.file = zout
        os.remove(self.filename)

        self._write_parquet('toc.parquet', new_toc)
        zout.close()
        os.rename('temp.zip', self.filename)

    def add_relation(self, df: pd.DataFrame):
        required_columns = ('file_1', 'file_2', 'relation_name')
        if not all(required_column in df.columns for required_column in required_columns):
            raise ValueError(f"{required_columns} must be in df columns")

        file_name = f"relation/{str(uuid4())}"
        self._write_parquet(file_name,df)

    def add_fragment(self, fragment: Union[CorpusFragment, pd.DataFrame], file_name=None):
        if isinstance(fragment, pd.DataFrame):
            fragment = CorpusFragment('', 0, fragment, {})

        if fragment.df.shape[0] == 0:
            return

        if fragment.filename not in self.indices:
            self.indices[fragment.filename] = 0
        else:
            self.indices[fragment.filename] += 1

        file_id = file_name if file_name is not None else str(uuid4())
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
        has_error = False
        toc = None
        
        if custom_toc is None:
            toc = pd.DataFrame(self.toc)
            toc.timestamp = toc.timestamp.astype('datetime64[s]')
            toc = toc.set_index('file_id')
        else:
            toc = custom_toc
        if 'toc.parquet' in self.file.namelist():
            self._replace_toc(toc)
        else:
            self._write_parquet('toc.parquet', toc)
            self.file.close()

    
        has_error = False

        if has_error:
            FileIO.write_pickle(self.toc, self.filename.parent / 'debug_toc_array.pickle')
            FileIO.write_pickle(toc, self.filename.parent / 'debug_toc_df.pickle')
            raise ValueError(
                'There was an issue with storaging TOC as a parquet dataframe. The corpus is finalized and readable. The pickled are stored in the same folder as the corpus, debug them and add toc.parquet in the zip folder manually')


    @staticmethod
    def collect_from_files(folder, target_file):
        folder = Path(folder)
        with zipfile.ZipFile(target_file, 'w', zipfile.ZIP_DEFLATED) as zp:
            for in_file_name in Query.folder(folder, '**/*'):
                if not in_file_name.is_file():
                    continue
                with open(in_file_name, 'rb') as in_file:
                    bytes = in_file.read()
                    relative_path = in_file_name.relative_to(folder)
                    zp.writestr(str(relative_path), bytes)
