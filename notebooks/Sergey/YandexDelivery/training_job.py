from typing import Iterable, List
from pathlib import Path
from notebooks.Sergey.YandexStorage.s3_yandex_helpers import S3YandexHandler
from tg.common import DataBundle
from tg.common.delivery.jobs import DeliverableJob
from tg.common.delivery.training.architecture import FileCacheTrainingEnvironment
from tg.grammar_ru.ml.components.training_task_factory import TaskFactory


class TrainingJob(DeliverableJob):
    def __init__(self, tasks: List[TaskFactory], project_name: str, bucket: str, ):
        super().__init__()
        self.tasks = tasks
        self.project_name = project_name
        self.bucket = bucket

    def get_name_and_version(self):
        return 'datasphere_testjob', 'v0'

    def run(self):
        for task in self.tasks:
            self._run_task(task)

    def _run_task(self, task: TaskFactory):
        if 'dataset' not in task.info:
            raise KeyError('task.info must contain dataset')
        dataset_path = self._download_dataset(task.info['dataset'])
        data = DataBundle.load(dataset_path)
        model_folder = Path("/models")
        env = FileCacheTrainingEnvironment(model_folder)
        task.run_with_environment(data, env)

    def _download_dataset(self, dataset_name: str) -> Path:
        s3path = f'datasphere/{self.project_name}/datasets/{dataset_name}'
        local_folder = Path('/datasets')
        S3YandexHandler.download_folder(self.bucket, s3path, local_folder)
        return local_folder
