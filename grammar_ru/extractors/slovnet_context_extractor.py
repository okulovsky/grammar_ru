import pandas as pd
from tg.common.ml import batched_training as bt
import numpy as np


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
        syntax_df = bundle.data_frames[self.dataframe_name]
        index_column = index_frame[self.index_column]

        new_rows = []

        for _, word_id in index_column.iteritems():
            parent_id = syntax_df[(syntax_df["word_id"] == word_id)]["parent_id"].item()
            shift = 0
            relative_id = word_id

            # Seeking for parent -> grandparent -> ...
            for _ in range(self.max_shift):
                shift += 1
                relative_id = syntax_df[(syntax_df["word_id"] == relative_id)]["parent_id"].item()
                if relative_id == -1:
                    break
                new_rows.append({'word_id': word_id, 'shift': shift, 'relative_word_id': relative_id})

            # Seeking for brothers, sisters ...
            for _, brother_row in syntax_df[(syntax_df["parent_id"] == parent_id)].iterrows():
                new_rows.append({'word_id': word_id, 'shift': 0, 'relative_word_id': brother_row['word_id']})

            def extract_child_rows(ids, iteration, max_shift, syntax_df):
                if iteration > max_shift:
                    return
                child_ids = []

                for i in ids:
                    for _, child_row in syntax_df[(syntax_df["parent_id"] == i)].iterrows():
                        child_ids.append(child_row['word_id'])

                for i in child_ids:
                    new_rows.append({'word_id': word_id, 'shift': -iteration, 'relative_word_id': i})

                extract_child_rows(child_ids, iteration + 1, max_shift, syntax_df)

            # Seeking for children and below ...
            extract_child_rows([word_id], 1, self.max_shift, syntax_df)

        return pd.DataFrame(new_rows)

    def get_name(self):
        return self.name
