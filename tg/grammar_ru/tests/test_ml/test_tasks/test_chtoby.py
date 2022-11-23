import pytest
import pandas as pd
import typing as tp

from tg.grammar_ru.common import Separator, DataBundle
from tg.grammar_ru.ml.tasks.train_index_builder.index_builders import ChtobyIndexBuilder


@pytest.fixture
def bundle() -> DataBundle:
    return Separator.build_bundle('что бы чтобы что бы затем что бы').src


@pytest.fixture
def preprocessed_df() -> pd.DataFrame:
    data = [
        (0, 'что бы', 1),
        (0, 'чтобы', 1),
        (0, 'что бы', 1),
        (0, 'затем', 0),
        (0, 'что бы', 1)
    ]

    return pd.DataFrame(data=data, columns=['sentence_id', 'word', 'is_target'])


def test_preprocessing_word(bundle: DataBundle) -> None:
    preprocessed = ChtobyIndexBuilder.preprocess(bundle.src)

    assert (preprocessed['word'] == ['что бы', 'чтобы', 'что бы', 'затем', 'что бы']).all()


def test_preprocessing_word_length(bundle: DataBundle) -> None:
    preprocessed = ChtobyIndexBuilder.preprocess(bundle.df)

    assert (preprocessed['word_length'] == [6, 5, 6, 5, 6]).all()


def test_get_targets(preprocessed_df: pd.DataFrame):
    builder = ChtobyIndexBuilder()
    targets = builder._get_targets(preprocessed_df)

    assert (targets == [True, True, True, False, True]).all()


def test_build_negative_from_positive(preprocessed_df: pd.DataFrame):
    builder = ChtobyIndexBuilder()
    assert False, print(builder._build_positive(preprocessed_df))


def test_build_train_index(df: pd.DataFrame, expected: pd.DataFrame):
    pass
