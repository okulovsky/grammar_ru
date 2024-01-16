from pathlib import Path

import pandas as pd
import torch

from tg.common import DataBundle
from tg.common.ml.batched_training import IndexedDataBundle


class MulticlassPredictionInterpreter:
    def interpret(self, input, labels, output):
        result = input["index"].copy()
        output = torch.softmax(output, dim=1)
        for i, c in enumerate(labels.columns):
            result["true_" + c] = labels[c]
            result["predicted_" + c] = output[:, i].tolist()
        return result


def read_train_bundle(bundle_path: Path) -> IndexedDataBundle:
    index_path = bundle_path / 'index.parquet'
    index = pd.read_parquet(index_path)
    db = DataBundle.load(bundle_path)

    return IndexedDataBundle(index, db)
