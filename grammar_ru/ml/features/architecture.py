from typing import *
import pandas as pd
from ...common import DataBundle

class Featurizer:
    def supports_update_on_column(self) -> Optional[str]:
        return None

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def as_enricher(self) -> 'Enricher':
        return EnricherOverFeaturizer(self, self.get_name(), self.supports_update_on_column())


class Enricher:
    def supports_updates_on_column(self) -> Optional[str]:
        return None

    def enrich(self, db: DataBundle) -> None:
        raise NotImplementedError()

    def get_df_name(self) -> str:
        raise NotImplementedError()

    def _update_enrich(self, on_column: str, old_bundle: Optional[DataBundle], new_bundle: DataBundle):
        if on_column is None or old_bundle is None:
            self.enrich(new_bundle)
            return

        name = self.get_df_name()
        old_df = old_bundle[name]
        id_name = old_df.index.name

        tmp_bundle = DataBundle(**new_bundle.data_frames)
        tmp_bundle['src'] = tmp_bundle.src.loc[tmp_bundle.src.updated]
        self.enrich(tmp_bundle)
        new_df = tmp_bundle[name]

        original_column = 'original_' + on_column
        old_index_to_new = new_bundle.src[[on_column, original_column]].drop_duplicates().set_index(on_column)
        old_index_to_new = old_index_to_new.loc[old_index_to_new[original_column] != -1]
        old_df = old_index_to_new.merge(
            old_bundle[name],
            left_on=original_column,
            right_index=True)
        old_df = old_df.drop(original_column, axis=1)

        df = pd.concat([old_df, new_df], axis=0)
        new_bundle[name] = df

    def update_enrich(self, old_bundle: Optional[DataBundle], new_bundle: DataBundle):
        self._update_enrich(
            self.supports_updates_on_column(),
            old_bundle,
            new_bundle
        )



class EnricherOverFeaturizer(Enricher):
    def __init__(self, featurizer: Featurizer, name: str, supports_update_on_column: Optional[str] = None):
        self.featurizer = featurizer
        self.name = name
        self._support_update_on_column = None

    def enrich(self, db: DataBundle) -> None:
        db.data_frames[self.name] = self.featurizer.featurize(db.data_frames['src'])

    def get_df_name(self) -> str:
        return self.name

    def supports_updates_on_column(self) -> Optional[str]:
        return self._support_update_on_column
