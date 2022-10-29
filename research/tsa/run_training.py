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


def execute_tasks(tasks):
    routine = create_sagemaker_routine('tsa', instance_type='ml.m5.xlarge')
    for t in tasks:
        routine.remote.execute(t, 'big', wait=False)


def run_local():
    tasks = autonamer.build_tasks(
        plain_network_mode = [ContextualNetworkType.LSTM],
        plain_net_size = [10],
        epoch_count = [5],
        batch_size = [1000],
        plain_context_length = [25],
        plain_context_left_shift = [0.5]
    )
    
    bundle = bt.DataBundle.load(Loc.data_cache_path/'bundles/tsa/bundles/toy')
    
    return {
        task.info['name']: task.run(bundle)
        for task in tasks
    }


def save_history(filename, history):
    import json
    with open(filename, "w") as f:
        json.dump(history, f)


def save_task(filename, task):
    import pickle
    with open(filename  + ".pickle", "wb") as f:
        pickle.dump(task['output']['training_task'], f)


if __name__ == '__main__':
    results = run_local()
    for task_name, task in results.items():
        pass
        # save_history(task_name, task['output']['history'])
        # save_task(task_name, task)
