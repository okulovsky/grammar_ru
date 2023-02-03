from tg.projects.create_sagemaker_routine import SagemakerRoutine
from tg.projects.alternative.alternative_task import AlternativeTrainingTask
from tg.common.delivery.sagemaker import Autonamer, download_and_open_sagemaker_result


def debug_run(in_docker = False):
    task = AlternativeTrainingTask('tsa-mini')
    task.settings.training_batch_limit = 1
    task.settings.evaluation_batch_limit = 1
    task.settings.epoch_count = 5
    routine = SagemakerRoutine(task)
    if not in_docker:
        result = routine.attached().execute()
        rs_display = [c['roc_auc_score_display'] for c in result['output']['history']]
        print(rs_display)
    else:
        routine.local().execute()


def remote_run(**kwargs):
    aut = Autonamer(AlternativeTrainingTask, 'att', common_arguments={'dataset': 'tsa-full'})
    tasks = aut.build_tasks(**kwargs)
    for task in tasks:
        routine = SagemakerRoutine(task)
        routine.remote().execute()



if __name__ == '__main__':
    pass
    #debug_run(False)
    #remote_run(epoch_count=[3], batch_size=[10000, 20000, 50000, 100000, 200000])
    #remote_run(hidden_size=[10, 20, 50, 100], learning_rate=[0.05, 0.02, 0.01])
    remote_run(hidden_size=[100], learning_rate=[0.01], context_length=[5, 9, 15, 21, 27], features=['p', None])


