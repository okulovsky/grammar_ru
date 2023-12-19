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
from tg.common.ml.batched_training import DataBundle, train_display_test_split, IndexedDataBundle
from tg.projects.alternative import DictionaryFilterer
from tg.common.ml import batched_training as bt


class AlternativeAlgorithm(NlpAlgorithm):
    def __init__(self):
        super().__init__()

    def create_db(self, text: str):
        text_db = Separator.build_bundle(text, [SnowballFeaturizer()])
        # text_db.index = text_db.src[['sentence_id', 'word']]
        return text_db

    def create_index(self, db: DataBundle, text: str) -> pd.DataFrame:
        # Read the source data
        src_df = db.src

        # Filter words ending with 'тся' or 'ться'
        regex = r'.*ться$|.*тся$'
        filtered_df = src_df[src_df['word'].str.contains(regex)]

        # Construct the index frame
        index_frame = filtered_df[['word_id', 'sentence_id']].copy()
        index_frame['label'] = -1  # Or use a placeholder value if needed
        index_frame['error'] = 0
        index_frame = index_frame.reset_index(drop=True)
        index_frame.index.name = 'sample_id'

        return index_frame

    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        model_path = Loc.model_path / 'alternative_task.pickle'
        idb = IndexedDataBundle(
            index_frame=index,
            bundle=db
        )
        with open(model_path, 'rb') as handle:
            output = pickle.load(handle)
            model = output
            # model = output['training_task']
            # batcher = output['batcher']
            # self.model = model
            # self.batcher = batcher
            # print('ok')

        return model.predict(idb)

