import requests
import torch
from typing import Optional
import pandas as pd
from tg.grammar_ru.algorithms.architecture import NlpAlgorithm
from tg.common.ml.batched_training import DataBundle
from tg.common.ml.batched_training import sandbox as bts
from tg.common.ml.batched_training import context as btc


class AlternativeAlgorithm(NlpAlgorithm):

    def __init__(self):
        super().__init__()

    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        task = bts.AlternativeTrainingTask2()
        task.settings.epoch_count = 1
        task.settings.batch_size = 20000
        task.settings.mini_epoch_count = 5
        task.optimizer_ctor.type = 'torch.optim:Adam'
        task.assembly_point.network_factory.network_type = btc.Dim3NetworkType.AlonAttention
        result = task.run(db)
        return result
