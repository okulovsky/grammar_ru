from typing import *
import pandas as pd
from tg.grammar_ru.common import DataBundle



class Featurizer:
    def get_frame_names(self) -> List[str]:
        raise NotImplementedError()

    def update_featurization(self, old_bundle: DataBundle, new_bundle: DataBundle):
        self.featurize(new_bundle)

    def featurize(self, db: DataBundle) -> None:
        raise NotImplementedError()

    def _simple_featurization_update_on_paragraph_id(self, old_bundle, new_bundle):
        frames = self.get_frame_names()
        if len(frames)!=1:
            raise ValueError('`_simple_featurization_update_on_paragraph_id` only works for featurizers providing exactly 1 frame')
        frame_name = frames[0]
        old_df = old_bundle[frame_name]
        index_name = old_df.index.name

        paragraphs_to_update = new_bundle.src.groupby('paragraph_id').updated.any().feed(lambda z: z.loc[z]).index
        src_to_update = new_bundle.src.loc[new_bundle.src.paragraph_id.isin(paragraphs_to_update)]

        tmp_bundle = DataBundle(**new_bundle.data_frames)
        tmp_bundle['src'] = src_to_update
        self.featurize(tmp_bundle)
        new_df = tmp_bundle[frame_name]

        original_column = 'original_' + index_name
        old_index_to_new = new_bundle.src[[index_name, original_column]].drop_duplicates().set_index(index_name)
        old_index_to_new = old_index_to_new.loc[old_index_to_new[original_column] != -1]
        old_df = old_index_to_new.merge(
            old_bundle[frame_name],
            left_on=original_column,
            right_index=True)
        old_df = old_df.drop(original_column, axis=1)

        df = pd.concat([old_df, new_df], axis=0)
        new_bundle[frame_name] = df

class SimpleFeaturizer(Featurizer):
    def __init__(self, frame_name: str, supports_updates = True):
        self.frame_name = frame_name
        self.supports_updates = supports_updates

    def _featurize_inner(self, db: DataBundle):
        raise NotImplementedError()

    def get_frame_names(self) -> List[str]:
        return [self.frame_name]

    def featurize(self, db: DataBundle) -> None:
        db[self.frame_name] = self._featurize_inner(db)

    def update_featurization(self, old_bundle: DataBundle, new_bundle: DataBundle):
        if self.supports_updates:
            self._simple_featurization_update_on_paragraph_id(old_bundle, new_bundle)
        else:
            self.featurize(new_bundle)

