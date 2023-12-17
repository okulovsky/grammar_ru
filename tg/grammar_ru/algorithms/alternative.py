from io import FileIO

import pickle
import requests
import torch
from typing import Optional
import pandas as pd
from yo_fluq_ds import *
from tg.grammar_ru import Separator
from tg.grammar_ru.features import SnowballFeaturizer
from tg.projects.alternative import DictionaryFilterer
from tg.projects.alternative import EndingNegativeSampler

from tg.common import Loc
from tg.grammar_ru.algorithms.architecture import NlpAlgorithm
from tg.common.ml.batched_training import DataBundle, train_display_test_split
from tg.projects.alternative import DictionaryFilterer
from tg.common.ml import batched_training as bt

class AlternativeAlgorithm(NlpAlgorithm):

    def __init__(self):
        super().__init__()

    def create_db(self, text: str):
        text_db = Separator.build_bundle(text, [SnowballFeaturizer()])
        return text_db

    def create_index(self, db: DataBundle, text: str):
        index = Separator.separate_string(text)
        good_words = set(FileIO.read_json(Loc.files_path / 'tsa-dict.json'))
        tsa_filter = DictionaryFilterer(good_words)
        index = tsa_filter.filter(index).index
        return index

    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        idb = bt.IndexedDataBundle(
            index_frame=index.to_frame(),
            bundle=db
        )
        model_path = Loc.model_path / 'alternative_task.pickle'
        with open(model_path, 'rb') as handle:
            model = pickle.load(handle)

        return model.predict(idb)

