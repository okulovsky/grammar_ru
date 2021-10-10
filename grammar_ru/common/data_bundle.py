from typing import *

import os
import pandas as pd

from pathlib import Path
from yo_fluq_ds import *


class DataBundle:
    def __init__(self, **data_frames: pd.DataFrame):
        self.data_frames = data_frames
        self.additional_information = Obj()

    @property
    def index_frame(self):
        return self.data_frames['index']

    @index_frame.setter
    def index_frame(self, value):
        self.data_frames['index'] = value

    def __getitem__(self, key):
        return self.data_frames[key]

    def __setitem__(self, key, value):
        self.data_frames[key] = value

    def __getattr__(self, key):
        return self.data_frames[key]

    def __contains__(self, item):
        return item in self.data_frames

    def __getstate__(self):
        return (self.data_frames, self.additional_information)


    def __setstate__(self, state):
        self.data_frames, self.additional_information = state

    def describe(self, index_sample_length = 5):
        return {
            key: dict(
                shape=list(value.shape),
                columns = list(value.columns),
                index_sample = list(value.index[:index_sample_length]))
            for key, value in self.data_frames.items()
        }
    def __str__(self):
        return self.describe().__str__()

    def __repr__(self):
        return self.describe().__repr__()


    @staticmethod
    def _read_bundle(path: Path):
        files = (Query
                 .folder(path)
                 .where(lambda z: z.name.endswith('.parquet'))
                 .to_list()
                 )
        data_frames = Query.en(files).to_dictionary(lambda z: z.name.split('.')[0], lambda z: pd.read_parquet(z))
        add_info = FileIO.read_pickle(path/'add_info.pkl')
        bundle = DataBundle(**data_frames)
        bundle.additional_information = add_info
        return bundle

    @staticmethod
    def load(inp: Union[Path, str]):
        """
        Loads bundle from filesystem
        """
        if isinstance(inp, str):
            inp = Path(inp)
        return DataBundle._read_bundle(inp)

    def save(self, folder: Union[str, Path]) -> None:
        if isinstance(folder, str):
            folder = Path(folder)
        os.makedirs(folder, exist_ok=True)
        for key, value in self.data_frames.items():
            value.to_parquet(folder.joinpath(key + '.parquet'))
        FileIO.write_pickle(self.additional_information, folder/'add_info.pkl')



