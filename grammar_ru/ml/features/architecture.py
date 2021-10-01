import pandas as pd
from ...common import DataBundle

class Featurizer:
    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def as_enricher(self) -> 'Enricher':
        return _EnricherOverFeaturizer(self, self.get_name())


class Enricher:
    def enrich(self, db: DataBundle) -> None:
        raise NotImplementedError()

    def get_df_name(self) -> str:
        raise NotImplementedError()


class _EnricherOverFeaturizer(Enricher):
    def __init__(self, featurizer: Featurizer, name: str):
        self.featurizer = featurizer
        self.name = name

    def enrich(self, db: DataBundle) -> None:
        db.data_frames[self.name] = self.featurizer.featurize(db.data_frames['src'])

    def get_df_name(self) -> str:
        return self.name
