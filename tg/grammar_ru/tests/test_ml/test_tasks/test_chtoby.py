from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import typing as tp

from tg.grammar_ru.common import Separator, DataBundle
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import ChtobyIndexBuilder


TEST_DF_PATH = Path(__file__).parent/'chtoby_test.parquet'

@pytest.fixture
def chtoby_df() -> pd.DataFrame:
    return pd.read_parquet(TEST_DF_PATH)


@pytest.fixture
def index_builder() -> ChtobyIndexBuilder:
    return ChtobyIndexBuilder()


def test_get_targets(chtoby_df: pd.DataFrame, index_builder: ChtobyIndexBuilder) -> None:
    targets = index_builder._get_targets(chtoby_df)

    assert not ('бы' in chtoby_df[targets]['word'])
    assert (chtoby_df[targets]['word'] == 'чтобы').sum() == 13
    assert (chtoby_df[targets]['word'] == 'что').sum() == 1


def test_build_negative_from_positive(chtoby_df: pd.DataFrame, index_builder: ChtobyIndexBuilder) -> None:
    _, negative = index_builder.build_train_index(chtoby_df)
    
    assert (negative['word'] == ['что', 'бы'] * 13 + ['чтобы']).all()
