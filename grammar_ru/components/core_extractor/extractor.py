from typing import *
from tg.common import Logger
from tg.common.ml import dft
from tg.common.ml import batched_training as bt
from .feature_description import features, FeatureSelection
import numpy as np
from yo_fluq_ds import KeyValuePair
from collections import OrderedDict


def log_transformer(df):
    return np.log(df+1)

def cont_transformer(df):
    return df.astype(float)

class CoreExtractorWrapper:
    def __init__(self, extractor, frame_name):
        self.extractor = extractor
        self.frame_name = frame_name
        self.disabled = False

class CoreExtractor(bt.Extractor):
    def __init__(self,
                 name: str = 'core',
                 join_column: str = 'word_id',
                 allow_list: Optional[List[str]]= None
                 ):
        self.name = name
        self.extractors = OrderedDict()
        for name, desc in features.items():
            self.extractors[name] = CoreExtractorWrapper(
                bt.PlainExtractor.build(name).join(desc.from_frame, join_column).apply(desc.to_transformer()),
                desc.from_frame
            )
        self.allow_list = None


    def fit(self, ibundle):
        for extractor in self.extractors.values():
            Logger.info(f'Fitting extractor {extractor.extractor.get_name()} in CoreExtractor')
            if self.allow_list is not None and extractor.extractor.get_name() not in self.allow_list:
                Logger.info('Skipped as not in allow list')
                extractor.disabled = True
                continue
            if extractor.frame_name not in ibundle.bundle.data_frames:
                Logger.info('Skipped as the corresponding frame is missing from the bundle')
                extractor.disabled = True
                continue
            extractor.disabled = False
            extractor.extractor.fit(ibundle)
            Logger.info('Success')


    def extract(self, ibundle):
        extractors = [c.extractor for c in self.extractors.values() if not c.disabled]
        df = bt.CombinedExtractor._run_extractors(ibundle, extractors)
        return df

    def get_name(self):
        return self.name


