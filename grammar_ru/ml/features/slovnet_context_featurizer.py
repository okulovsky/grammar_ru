from .architecture import *
from grammar_ru.ml.features import SlovnetFeaturizer
from yo_fluq_ds import *


class SlovnetContextFeaturizer(Featurizer):
    def __init__(self, max_shift=3):
        self.max_shift = max_shift

    def get_name(self):
        return 'slovnet_context'

    def featurize(self, df):
        prepared_df = df
        if "syntax_parent_id" not in df.columns:
            prepared_df = SlovnetFeaturizer().featurize(df)

        return self._extract_context(prepared_df)

    def _extract_context(self, df: pd.DataFrame):
        parent_df = df.reset_index().sort_values(by=["syntax_parent_id", "word_id"]).set_index([
            "syntax_parent_id", "word_id"])

        new_rows = []

        for word_id in df.index:
            parent_id = df.loc[word_id, "syntax_parent_id"]
            shift = 0
            relative_id = word_id

            # Seeking for parent -> grandparent -> ...
            for _ in range(self.max_shift):
                shift += 1
                relative_id = df.loc[relative_id, "syntax_parent_id"]
                if relative_id == -1:
                    break
                new_rows.append({'word_id': word_id, 'shift': shift, 'relative_word_id': relative_id})

            # Seeking for brothers, sisters ...
            for idx in parent_df.loc[parent_id].index if parent_id in parent_df.index else []:
                new_rows.append({'word_id': word_id, 'shift': 0, 'relative_word_id': idx})

            ids = [word_id]
            for i in range(1, self.max_shift + 1):
                child_ids = []

                for par in ids:
                    for idx in parent_df.loc[par].index if par in parent_df.index else []:
                        child_ids.append(idx)

                for child in child_ids:
                    new_rows.append({'word_id': word_id, 'shift': -i, 'relative_word_id': child})

                ids = child_ids

        return pd.DataFrame(new_rows).set_index("word_id")
