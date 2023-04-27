import pickle
import pandas as pd
import numpy as np

from tg.grammar_ru import features
from tg.grammar_ru.common import Loc, Separator, DataBundle
from tg.grammar_ru.algorithms import NlpAlgorithm
from tg.common.ml import batched_training as bt
from tg.projects.punct.models import PunctNetworkNavec, punct_network_factory_navec


class PunctNlpAlgorithm(NlpAlgorithm):
    def __init__(self, model, batcher, path_to_navec_vocab):
        self.featurizer = features.PyMorphyFeaturizer()
        self.model = model
        self.path_to_navec_vocab = path_to_navec_vocab

        self._vocab = pd.read_parquet(path_to_navec_vocab)
        self._batcher = batcher
        self._filter_batcher_extractors()

    def _filter_batcher_extractors(self):
        allowed_extractors = ['features', 'navec', 'label']
        filtered_extractors = list(filter(lambda e: e.name in allowed_extractors, self._batcher.extractors))
        self._batcher.extractors = filtered_extractors

    def _create_idb(self, db: DataBundle) -> bt.IndexedDataBundle:
        index = db.src.copy()
        index['label'] = np.full(index.shape[0], 0)
        index['target_word'] = np.full(index.shape[0], 'no')
        index.loc[index.shape[0] - 3, 'target_word'] = '-'  # FIXME
        index.loc[index.shape[0] - 2, 'target_word'] = ','  
        index.loc[index.shape[0] - 1, 'target_word'] = ':'  

        idb = bt.IndexedDataBundle(
            index_frame=index,
            bundle=db,
        )

        return idb

    def _add_feature_frames(self, db: DataBundle):
        self.featurizer.featurize(db)
        db['sample_to_navec'] = self._vocab

    def _extract_features(self, db: DataBundle) -> DataBundle:
        db.src.index.name = 'sample_id'
        db.src['is_target'] = np.full(db.src.shape[0], True)
        self._add_feature_frames(db)
        ibundle = self._create_idb(db)

        sequence_length = db.src.shape[0]
        extracted = self._batcher.get_batch(sequence_length, ibundle, 0)

        return extracted

    def _get_predicted_symbols(self, db: DataBundle):
        features = self._extract_features(db)
        pred_df = self.model.predict(features)
        predicted_columns = pred_df.columns[pred_df.columns.str.startswith('predicted')]
        symbols = predicted_columns.map(lambda x: x.split('_')[1]).values

        predictions = np.argmax(pred_df[predicted_columns].values, axis=1)

        return np.repeat(symbols.reshape(1, -1), len(predictions), axis=0)[range(len(predictions)),predictions]

    def _run_inner(self, db: DataBundle, index: pd.Index) -> pd.DataFrame:
        df = db.src.loc[index]
        result = pd.DataFrame({}, index=df.index)

        to_check = (df.word_type == 'ru')
        to_check[df.shape[0] - 1] = False
        result[NlpAlgorithm.Error] = False
        predicted_symbols = self._get_predicted_symbols(db)

        shifted_words = df.word.shift(-1)
        wrong = (shifted_words != predicted_symbols) & (predicted_symbols != 'no')
        wrong = wrong | (predicted_symbols == 'no') & (shifted_words.isin((',', ':', '—')))
        wrong = wrong[to_check]

        result.loc[to_check, NlpAlgorithm.Error] = wrong
        result.loc[result[NlpAlgorithm.Error], NlpAlgorithm.Suggest] = predicted_symbols[to_check & wrong]

        result[NlpAlgorithm.ErrorType] = 'syntax'  # TODO: add error type

        return result


class PunctModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'punct_network_factory_navec':
            from tg.projects.punct.models import punct_network_factory_navec
            return punct_network_factory_navec
        if name == 'PunctNetworkNavec':
            from tg.projects.punct.models import PunctNetworkNavec
            return PunctNetworkNavec

        return super().find_class(module, name)
