from .loc import Loc
from .separator import Separator, Symbols
from .df_viewer import DfViewer
from .data_bundle import DataBundle
from .misc import sample_table


import zipfile
from io import BytesIO
import os
import pandas

def _save_bundle_as_zip(bundle, fname):
    with zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED) as file:
        for name, data in bundle.data_frames.items():
            bytes = BytesIO()
            data.to_parquet(bytes)
            file.writestr(name + '.parquet', bytes.getbuffer())

def _load_zip(fname):
    with zipfile.ZipFile(fname, 'r') as file:
        bundle = DataBundle()
        for f in file.filelist:
            name = f.filename
            parq = '.parquet'
            if name.endswith(parq):
                buffer = BytesIO(file.read(name))
                df = pandas.read_parquet(buffer)
                bundle[name.replace(parq,'')] = df
    return bundle

_old_load = DataBundle.load

def load_bundle_as_zip(fname):
    if os.path.isdir(fname):
        return _old_load(fname)
    elif os.path.isfile(fname):
        return  _load_zip(fname)
    else:
        raise ValueError(f'{fname} is neither file nor directory')

DataBundle.load = load_bundle_as_zip
DataBundle.save_as_zip = _save_bundle_as_zip




