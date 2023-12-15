from io import FileIO

import requests
import torch
from typing import Optional
import pandas as pd

from tg.common import Loc
from tg.grammar_ru.algorithms.architecture import NlpAlgorithm
from tg.common.ml.batched_training import DataBundle, train_display_test_split
from tg.projects.alternative import DictionaryFilterer


class AlternativeAlgorithm(NlpAlgorithm):

    def __init__(self):
        super().__init__()

    def index_db(self, db: DataBundle):
        index_frame = db.data_frames['src']
        index_frame = index_frame.loc[index_frame.is_target][['word_id', 'sentence_id', 'label']].reset_index(drop=True)
        index_frame.index.name = 'sample_id'
        #index_frame['split'] = train_display_test_split(index_frame)



    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        model_path = Loc.model_path / 'alternative_model.zip'
        model = torch.load(model_path)
        model.eval()
        return model.predict(db)

