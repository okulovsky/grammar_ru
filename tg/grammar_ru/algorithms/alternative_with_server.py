import requests
import torch
from typing import Optional
import pandas as pd
from tg.grammar_ru.algorithms.architecture import NlpAlgorithm
from tg.common.ml.batched_training import DataBundle


class AlternativeAlgorithmWithServer(NlpAlgorithm):
    def __init__(self, model_server_url: str):
        super().__init__()
        self.model_server_url = model_server_url
        self.model = self._load_model_from_server()

    def _load_model_from_server(self):
        response = requests.get(self.model_server_url)
        response.raise_for_status()
        model_data = response.content
        model = torch.load(model_data)
        return model

    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        input_data = db.data
        output_data = self.model(input_data)
        output_df = pd.DataFrame(output_data, index=index)
        return output_df