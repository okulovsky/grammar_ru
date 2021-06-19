import pandas as pd
from tg.common.ml import batched_training as bt
from yo_fluq_ds import KeyValuePair


class ContextExtractor(bt.Extractor):
    """
    This extractor should be applied to dataframes created by NatashaSyntaxAnalyzer.
    Extracts contexts of syntax tree by calculating shifts: brothers, parents, children...
    """

    """
    Args:
        max_shift: Specifies the maximum shift relative to the selected object.
        custom_dataframe_name: Specifies bundle name of dataframe that was created by NatashaSyntaxAnalyzer.
        custom_index_name: Specifies column name in index row.
    """

    def __init__(self, name: str, max_shift=2, custom_dataframe_name="syntax", custom_index_column="word_id"):
        self.name = name
        self.max_shift = max_shift
        self.dataframe_name = custom_dataframe_name
        self.index_column = custom_index_column

    def fit(self, bundle: bt.DataBundle):
        pass

    def extract(self, index_frame: pd.DataFrame, bundle: bt.DataBundle):
        syntax_df = bundle.data_frames[self.dataframe_name].set_index(self.index_column)
        parent_df = bundle.data_frames[self.dataframe_name].sort_values(by=["parent_id", self.index_column]).set_index(["parent_id", self.index_column])

        new_rows = []

        for word_id in index_frame.index:
            parent_id = syntax_df.at[word_id, "parent_id"]
            shift = 0
            relative_id = word_id

            # Seeking for parent -> grandparent -> ...
            for _ in range(self.max_shift):
                shift += 1
                relative_id = syntax_df.at[relative_id, "parent_id"]
                if relative_id == -1:
                    break
                new_rows.append({'word_id': word_id, 'shift': shift, 'relative_word_id': relative_id})

            # Seeking for brothers, sisters ...
            for idx, brother_row in parent_df.loc[parent_id].iterrows() if parent_id in parent_df.index else []:
                new_rows.append({'word_id': word_id, 'shift': 0, 'relative_word_id': idx})
            
            ids = [word_id]
            for i in range(1, self.max_shift + 1):
                child_ids = []

                for par in ids:
                    for idx, child_row in parent_df.loc[par].iterrows() if par in parent_df.index else []:
                        child_ids.append(idx)
                
                for child in child_ids:
                    new_rows.append({'word_id': word_id, 'shift': -i, 'relative_word_id': child})
                
                ids = child_ids

        return KeyValuePair(self.name, pd.DataFrame(new_rows))

    def get_name(self):
        return self.name
