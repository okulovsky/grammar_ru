from typing import *
from grammar_ru.training import amenities
from tg.common.ml import batched_training as bt
from sklearn.metrics import roc_auc_score
from functools import partial
import pandas as pd

class ModelSettings:
    def __init__(self,
                 batch_size: int,
                 network_sizes: List[int],
                 learning_rate: float,
                 ):
        self.batch_size = batch_size
        self.article_network_sizes = network_sizes
        self.learning_rate = learning_rate

    def to_name(self):
        return (
            'REP-' +
            'LR' + str(self.learning_rate)+"-" +
            'NT' + '_'.join([str(c) for c in self.article_network_sizes]) + '-'
        )



class Experiment(bt.BatchedTrainingTask):
    def __init__(self, train_settings, settings: ModelSettings):
        super(Experiment, self).__init__(settings = train_settings, late_initialization=Experiment.init)
        self.model_settings = settings
        self.info['name'] = self.model_settings.to_name()
        self.splitter = bt.CompositionSplit(bt.FoldSplitter(1,0.2), bt.FoldSplitter(1, 0.2, 'display', decorate=True))
        self.metric_pool = bt.MetricPool().add_sklearn(roc_auc_score)


    def correct_bundle(self, db):
        idf = db.index_frame
        idf = idf.merge(db.data_frames['src'][['sentence_id', 'paragraph_id']], left_on='word_id', right_index=True)

        another_ids = db.data_frames['src'][['sentence_id', 'paragraph_id']].rename(
            columns=dict(sentence_id='another_sentence_id', paragraph_id = 'another_paragraph_id'))

        idf = idf.merge(another_ids, left_on='another_word_id', right_index=True)

        idf['delta_word_id'] = idf.word_id - idf.another_word_id
        idf['delta_sentence_id'] = idf.sentence_id - idf.another_sentence_id
        idf['delta_paragraph_id'] = idf.paragraph_id - idf.another_paragraph_id

        distance_df = idf[['word_id','delta_word_id','delta_sentence_id','delta_paragraph_id']].set_index('word_id')
        db.data_frames['distance'] = distance_df

        label = db.data_frames['toc'].source == 'books'
        idf = idf.merge(label.to_frame('label'), left_on='file_id', right_index=True)
        idf['priority'] = bt.PriorityRandomBatcherStrategy.make_priorities_for_even_representation(idf, 'label')

        db.index_frame = idf

    def get_extractors(self):
        cf = amenities.create_feature_transformer
        extractors = []
        for field in ['word_id','another_word_id']:
            for frame in ['capitalization', 'local_freq', 'pymorphy', 'frequencies']:
                extractors.append(bt.DirectExtractor(frame+'_'+field, cf(), frame, field))
        extractors.append(bt.DirectExtractor('distance',cf(), 'distance', 'word_id'))
        features = bt.CombinedExtractor('features',extractors)
        labels = bt.IndexExtractor('labels',None, 'label')
        return [features, labels]

    def get_model_handler(self):
        return amenities.TorchModelHandler(
            partial(
                amenities.LayeredNetworkWithExtraction,
                source_frame='features',
                dst_frame='labels',
                size=self.model_settings.article_network_sizes),
            self.model_settings.learning_rate
        )


    def init(self, bundle: bt.DataBundle, env: bt.TrainingEnvironment):
        env.log('Correcting bundle')
        self.correct_bundle(bundle)

        if 'priority' in bundle.index_frame.columns:
            strategy = bt.PriorityRandomBatcherStrategy('priority')
        else:
            strategy = None

        extractors = self.get_extractors()
        self.batcher = bt.Batcher(self.model_settings.batch_size, extractors, strategy)
        self.model_handler = self.get_model_handler()
        env.log('Training initialization complete')






















