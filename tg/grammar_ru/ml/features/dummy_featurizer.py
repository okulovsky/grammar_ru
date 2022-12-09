from ....grammar_ru.common import Loc
from .architecture import Featurizer, DataBundle

from typing import List


class DummyFeaturizer(Featurizer):
    def __init__(self, unnecessary_keys: List[str]) -> None:
        self.unnecessary_keys = unnecessary_keys

    def featurize(self, db: DataBundle) -> None:
        for key in self.unnecessary_keys:
            db.data_frames.pop(key)
