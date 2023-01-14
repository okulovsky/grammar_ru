import torch
from sklearn.metrics import roc_auc_score
from .....common.ml import batched_training as bt
from .....common.ml.batched_training.torch.networks.simple_networks import FullyConnectedNetwork
from ...components.core_extractor.extractor import CoreExtractor
from ...components.plain_context_builder import PlainContextBuilder
from ...components.training_task_factory import TaskFactory, Conventions
from ...components.contextual_binding import ContextualBinding, ContextualNetworkType


def get_binary_label_extractor():
    label_extractor = (bt.PlainExtractor
                       .build(Conventions.LabelFrame)
                       .index()
                       .apply(take_columns=['label'], transformer=None))
    return label_extractor


def get_plain_context(sentence_id_column_name, word_id_column_name, context_name):
    pcb = PlainContextBuilder(include_zero_offset=True, left_to_right_contexts_proportion=0.5)
    pcb.sentence_id_column_name = sentence_id_column_name
    pcb.word_id_column_name = word_id_column_name
    plain_context = ContextualBinding(
        name=context_name,
        context_length=3,
        network_type=ContextualNetworkType.Plain,
        hidden_size=[30],
        context_builder=pcb,
        extractor=CoreExtractor(join_column='another_word_id'),
        debug=False
    )
    return plain_context


class MyCandidateNetwork(torch.nn.Module):
    def __init__(self, batch, pronoun_head, candidate_head, tail_network_size):
        super(MyCandidateNetwork, self).__init__()
        self.pronoun_head = pronoun_head
        self.candidate_head = candidate_head
        heads_output = self._head_step(batch)
        self.fully_connected = FullyConnectedNetwork([tail_network_size, 1],
                                                     heads_output)

    def _head_step(self, batch):
        pronoun_out = self.pronoun_head(batch)
        candidate_out = self.candidate_head(batch)
        concat = torch.cat([pronoun_out, candidate_out], dim=1)
        return concat

    def forward(self, batch):
        heads_output = self._head_step(batch)
        return self.fully_connected(heads_output)


class MyCandidateNetworkFactory:
    def __init__(self, pronoun_pc, candidate_pc, tail_network_size):
        self.pronoun_pc = pronoun_pc
        self.candidate_pc = candidate_pc
        self.tail_network_size = tail_network_size

    def create(self, task, batch):
        pronoun_head = (self.pronoun_pc
                        .create_network_factory(task=None, input=None)
                        .create_network(task=None, input=batch))
        candidate_head = (self.candidate_pc
                          .create_network_factory(task=None, input=None)
                          .create_network(task=None, input=batch))
        return MyCandidateNetwork(batch, pronoun_head, candidate_head,
                                  self.tail_network_size)


class AntecedentCandidateTask(TaskFactory):
    def __init__(self):
        super(AntecedentCandidateTask, self).__init__()
        self.pronoun_pc = get_plain_context('pronoun_sentence_id',
                                            'pronoun_word_id',
                                            'pronoun_context')
        self.candidate_pc = get_plain_context('candidate_sentence_id',
                                              'candidate_word_id',
                                              'candidate_context')

    def _create_network(self, task, batch):

        network_factory = MyCandidateNetworkFactory(self.pronoun_pc,
                                                    self.candidate_pc,
                                                    50)
        return network_factory.create(None, batch)

    def create_task(self, data, env):
        metrics = bt.MetricPool().add_sklearn(roc_auc_score)
        self.instantiate_default_task(epoch_count=50, batch_size=1000,
                                      mini_batch_size=None,
                                      metric_pool=metrics)

        extractors = [
            self.pronoun_pc.create_extractor(task=None, bundle=data),
            self.candidate_pc.create_extractor(task=None, bundle=data),
            get_binary_label_extractor()
        ]
        self.setup_batcher(data, extractors)
        self.setup_model(self._create_network, learning_rate=1)
