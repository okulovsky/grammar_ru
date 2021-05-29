from typing import *
import pandas as pd

WordCoordinates = ['word', 'word_id', 'sentence_id', 'word_index']


def ensure_df_contains(columns: List[str], df: pd.DataFrame):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column `{column}` not in dataframe")
