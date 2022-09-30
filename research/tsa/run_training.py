from typing import *
from tg.amenities import create_sagemaker_routine
from tg.grammar_ru.ml.components import GrammarMirrorSettings, ContextualNetworkType
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import torch as btt
from tg.common.ml.batched_training import mirrors as btm
from tg.common import Loc
from sklearn.metrics import roc_auc_score
from yo_fluq_ds import Query
from enum import Enum
from tg.common.delivery.training import Autonamer


features = {
    'P': 'pymorphy',
    'M': 'slovnet_morph',
    'S': 'slovnet_syntax',
    'F': 'syntax_fixes',
    'T': 'syntax_stats'
}


def build_task(
        epoch_count:int = 50,
        batch_size: int = 20000,
        mini_batch_size: int = 200,
        mini_epoch_count: int = 4,
        learning_rate: float = 0.1,
        plain_context_length: int = 10,
        plain_context_left_shift: float = 0.5,
        plain_net_size = [20],
        plain_network_mode = ContextualNetworkType.Plain,
        plain_context_reverse = False,
        feature_allow_list = None
):
    train_settings = bt.TrainingSettings(
        epoch_count=epoch_count,
        batch_size = batch_size,
        mini_batch_size = mini_batch_size,
        mini_epoch_count = mini_epoch_count
    )
    torch_settings = btt.TorchTrainingSettings(
        btt.OptimizerConstructor('torch.optim:SGD', lr=learning_rate)
    )
    mirror_settings = GrammarMirrorSettings()
    mirror_settings.plain_context.context_builder.left_to_right_contexts_proportion = plain_context_left_shift
    mirror_settings.plain_context.context_length = plain_context_length
    mirror_settings.plain_context.hidden_size = plain_net_size
    mirror_settings.plain_context.network_type = plain_network_mode
    mirror_settings.plain_context.reverse_order_in_lstm = plain_context_reverse

    if feature_allow_list is not None:
        mirror_settings.plain_context.extractor.allow_list = [features[l] for l in feature_allow_list]

    task = btm.MirrorTrainingTask(
        train_settings,
        torch_settings,
        mirror_settings,
        bt.MetricPool().add_sklearn(roc_auc_score),
    )
    task.info['name'] = 'TSAG-'
    return task

autonamer = Autonamer(build_task)


def run_local():
    task = build_task(plain_network_mode=ContextualNetworkType.Plain)
    task.settings.batch_size = 1000
    task.settings.training_batch_limit = 2
    task.settings.evaluation_batch_limit = 2
    bundle = bt.DataBundle.load(Loc.data_cache_path/'bundles/tsa/bundles/toy')
    print(task.info['name'])
    task.run(bundle)
    exit(0)

def execute_tasks(tasks):
    routine = create_sagemaker_routine('tsa', instance_type='ml.m5.xlarge')
    for t in tasks:
        routine.remote.execute(t, 'big', wait=False)




if __name__ == '__main__':
    #run_local()

    tasks = autonamer.build_tasks(
        #TODO: plain_network_type = ContextualNetworkType.Attention
    )
    # print(tasks[0].info['name'])
    run_local()

