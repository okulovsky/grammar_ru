from .architecture import *
from grammar_ru.ml.features import SlovnetFeaturizer
from yo_fluq_ds import *


class SlovnetContextFeaturizer(Featurizer):
    _wordId = "word_id"
    _parentId = "syntax_parent_id"
    _shift = "shift"
    _relativeWordId = "relative_word_id"

    def __init__(self, max_shift=3):
        self.max_shift = max_shift

    def get_name(self):
        return 'slovnet_context'

    def featurize(self, df):
        prepared_df = df
        if self._parentId not in df.columns:
            prepared_df = SlovnetFeaturizer().featurize(df)

        return self._extract_context(prepared_df)

    """
        Non-optimized (iterative approach is used)
        Can be optimized by some vectorization operations.
        (However, we are unable to get rid of iterative approach everywhere in this function)
    """
    def _extract_context(self, df: pd.DataFrame):
        parent_df = df.reset_index().sort_values(by=[self._parentId, self._wordId]).set_index([
            self._parentId, self._wordId])

        new_rows = []

        for word_id in df.index:
            parent_id = df.loc[word_id, self._parentId]
            shift = 0
            relative_id = word_id

            # Looking for parent -> grandparent -> ...
            for _ in range(self.max_shift):
                shift += 1
                relative_id = df.loc[relative_id, self._parentId]
                if relative_id == -1:
                    break
                new_rows.append({self._wordId: word_id, self._shift: shift, self._relativeWordId: relative_id})

            # Looking for brothers, sisters ...
            for idx in parent_df.loc[parent_id].index if parent_id in parent_df.index else []:
                new_rows.append({self._wordId: word_id, self._shift: 0, self._relativeWordId: idx})

            # Looking for children...
            ids = [word_id]
            for i in range(1, self.max_shift + 1):
                child_ids = []

                for par in ids:
                    for idx in parent_df.loc[par].index if par in parent_df.index else []:
                        child_ids.append(idx)

                for child in child_ids:
                    new_rows.append({self._wordId: word_id, self._shift: -i, self._relativeWordId: child})

                ids = child_ids

        return pd.DataFrame(new_rows).set_index(self._wordId)
