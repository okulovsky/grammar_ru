from typing import *
import pandas as pd
from grammar_ru.common import validations


def make_dataframe_from_nerus() -> pd.DataFrame:
    pass


def create_chunks_from_dataframe(df: pd.DataFrame) -> List[List[str]]:
    validations.ensure_df_contains(validations.WordCoordinates)
    pass
