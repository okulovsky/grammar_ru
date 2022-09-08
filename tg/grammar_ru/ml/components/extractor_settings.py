from ....common.ml import dft
from ....common.ml.batched_training import mirrors as btm
from ....common.ml.batched_training import torch as btt
from ....common.ml import batched_training as bt
from .plain_context_builder import PlainContextBuilder
from .core_extractor.extractor import CoreExtractor

class GrammarMirrorSettings(btm.MirrorSettings):
    def __init__(self,
                 debug = False,
                 apply_transformer_to_labels = False

                 ):
        self.plain_context = btm.ContextualBinding(
            'plain_context',
            10,
            btm.ContextualNetworkType.Plain,
            [30],
            PlainContextBuilder(True, 0),
            CoreExtractor(join_column='another_word_id'),
            debug
        )

        fields = [v for v in self.__dict__.values() if isinstance(v, btm.ExtractorNetworkBinding)]
        labels_transformer = None if not apply_transformer_to_labels else dft.DataFrameTransformerFactory.default_factory()

        super(GrammarMirrorSettings, self).__init__(
            [btt.FullyConnectedNetwork.Factory([], output='label').prepend_extraction(['plain_context'])],
            bt.PlainExtractor.build(btt.Conventions.LabelFrame).apply(take_columns='label', transformer=labels_transformer),
            fields
        )
