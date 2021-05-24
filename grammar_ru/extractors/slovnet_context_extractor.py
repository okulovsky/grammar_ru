import pandas as pd
from tg.common.ml import dft
from tg.common.ml import batched_training as bt
import numpy as np


def add_new_shift(word_id, df):


class ContextTransformer(dft.DataFrameTransformer):
    """
    This transformer should be applied to dataframes created by NatashaSyntaxAnalyzer.
    Extracts contexts of syntax tree by calculating shifts: brothers, parents, children...
    """

    def __init__(self, selector=None, max_shift=2):
        """

        Args:
            selector: A function that returns either true or false depending on the object properties. Default value falls back to always return True.
            max_shift: Specifies max shift relative to the selected object.
        """
        if not selector:
            def default_selector(_): return True
            selector = default_selector
        self.selector = selector
        self.max_shift = max_shift

    def fit(self, df):
        pass

    def transform(self, df):
        if "shift" in df.columns or "relative_word_id" in df.columns:
            raise ValueError("Dataframe already contains either 'shift' or 'relative_word_id' column")

        result = pd.DataFrame(df)
        result["shift"] = 1
        result[result["parent_id"].isna()]["shift"] = np.nan
        result["relative_word_id"] = result["parent_id"]

        return something(df)


# TODO: I need to make extractor out of this.
# An extractor is a transformer + selector of columns
# In our case we also have a selector of rows.
# Firstly, we need to find row using selector. Secondly, we need to process it (find it's parents, brothers, children)
# Thirdly, we just return found values
