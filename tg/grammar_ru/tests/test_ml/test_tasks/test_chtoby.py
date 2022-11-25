import pytest
import pandas as pd
import numpy as np
import typing as tp

from tg.grammar_ru.common import Separator, DataBundle
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import ChtobyIndexBuilder


@pytest.fixture
def bundle() -> DataBundle:
    return Separator.build_bundle('что бы чтобы что бы затем что бы')


@pytest.fixture
def preprocessed_df() -> pd.DataFrame:
    data = [
        (1, 'что бы', True),
        (1, 'чтобы', True),
        (2, 'что бы', True),
        (3, 'затем', False),
        (4, 'что бы', True)
    ]

    return pd.DataFrame(
        data=data,
        columns=['sentence_id', 'word', 'is_target'],
        index=np.arange(len(data)),
    )


def test_preprocessing_word(bundle: DataBundle) -> None:
    preprocessed = ChtobyIndexBuilder.preprocess(bundle.src)

    assert (preprocessed['word'] == ['что бы', 'чтобы', 'что бы', 'затем', 'что бы']).all()


def test_preprocessing_word_length(bundle: DataBundle) -> None:
    preprocessed = ChtobyIndexBuilder.preprocess(bundle.src)

    assert (preprocessed['word_length'] == [6, 5, 6, 5, 6]).all()


def test_get_targets(preprocessed_df: pd.DataFrame) -> None:
    builder = ChtobyIndexBuilder()
    targets = builder._get_targets(preprocessed_df)

    assert (targets == [True, True, True, False, True]).all()


def test_build_negative_from_positive(preprocessed_df: pd.DataFrame):
    builder = ChtobyIndexBuilder()
    expected = [
        (1, 'чтобы', True, 1),
        (1, 'что бы', True, 1),
        (2, 'чтобы', True, 1),
        (3, 'затем', False, 1),
        (4, 'чтобы', True, 1)
    ]

    assert (builder._build_negative_from_positive(preprocessed_df).to_numpy() == np.array(expected, dtype=object)).all()
