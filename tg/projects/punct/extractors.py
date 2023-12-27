import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from .context_builder import PunctContextBuilder

from ...common.ml import batched_training as bt
from ...common.ml.batched_training import factories as btf
from ...common.ml.batched_training import context as btc
from ...common.ml import dft
from ...grammar_ru.components import PlainContextBuilder


class DataFrameLabelBinarizer():
    def __init__(self, column_name: str):
        self.column_name = column_name
        self.encoder = LabelBinarizer()

    def fit(self, df: pd.DataFrame) -> 'DataFrameLabelBinarizer':
        self.encoder.fit(df[self.column_name])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_columns = self.encoder.transform(df[self.column_name])
        if transformed_columns.shape[1] == 1:
            result = pd.DataFrame(index=df.index, columns=list(self.encoder.classes_[0]), data=transformed_columns)
        else:
            result = pd.DataFrame(index=df.index, columns=self.encoder.classes_, data=transformed_columns)

        return result


def create_label_extractor():
    label_extractor = (
        bt.PlainExtractor
        .build(btf.Conventions.LabelFrame)
        .index()
        .apply(take_columns='target_word', transformer=DataFrameLabelBinarizer('target_word'))
    )

    return label_extractor


def create_vocab_extractor():
    vocab_extractor = (
        bt.PlainExtractor
        .build('vocab')
        .index()
        .join('sample_to_vocab', on_columns='word_id')
        .apply()
    )

    return vocab_extractor


def create_navec_extractor():
    navec_extractor = (
        bt.PlainExtractor
        .build('navec')
        .index()
        .join('sample_to_navec', on_columns='word_id')
        .apply()
    )

    return navec_extractor


def create_pymorphy_extractor():
    pymorphy_extractor = (
        bt
        .PlainExtractor
        .build('pymorphy')
        .index()
        .join('pymorphy', on_columns='another_word_id')
        .apply(
            transformer=dft.DataFrameTransformerFactory.default_factory(),
        )
    )

    return pymorphy_extractor


def create_assembly_point(extractor, context_builder, context_length=20):
    ap = btc.ContextualAssemblyPoint(
        name='features',
        context_builder=context_builder,
        extractor=extractor,
        context_length=context_length
    )
    ap.reduction_type = ap.reduction_type.Dim3Folded
    ap.dim_3_network_factory.network_type = btc.Dim3NetworkType.LSTM

    return ap


def create_context_extractor(train=True):
    if train:
        context_builder = PunctContextBuilder(
            include_zero_offset=True,
            left_to_right_contexts_proportion=0.5
        )
    else:
        context_builder = PlainContextBuilder(
            include_zero_offset=True,
            left_to_right_contexts_proportion=0.5
        )

    pymorphy_extractor = create_pymorphy_extractor()
    assembly_point = create_assembly_point(pymorphy_extractor, context_builder)

    return assembly_point.create_extractor()
