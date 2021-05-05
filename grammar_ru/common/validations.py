from typing import *
import pandas as pd


def validate_df_contains(columns: List[str], df: pd.DataFrame):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column `{column}` not in dataframe")
