from typing import Iterable, List
from pathlib import Path
import datetime
import tarfile
import os.path
import shutil
from notebooks.Sergey.YandexStorage.s3_yandex_helpers import S3YandexHandler
from tg.common import DataBundle
from tg.common.delivery.jobs import DeliverableJob
from tg.common.delivery.training.architecture import FileCacheTrainingEnvironment
from tg.grammar_ru.ml.components.training_task_factory import TaskFactory
from tg.grammar_ru.common.loc import Loc


class TrainingJob(DeliverableJob):
    def __init__(self, tasks: List[TaskFactory], project_name: str, bucket: str):
        super().__init__()
        self.tasks = tasks
        self.project_name = project_name
        self.bucket = bucket

    def get_name_and_version(self):
        return 'datasphere_testjob', 'v1'

    def run(self):
        for task in self.tasks:
            self._run_task(task)

    def _try_run(self, task, data, env):
        try:
            task.run_with_environment(data, env)
            return True
        except Exception as e:
            S3YandexHandler.save_to_file(self.bucket,
                                         s3_path=f"datasphere/{self.project_name}/exceptions/{task.name}.txt",
                                         content=str(e))
            print(f'Exception in {task.name} loaded to DataSphere')
            return False

    def _run_task(self, task: TaskFactory):
        if 'dataset' not in task.info:
            raise KeyError('task.info must contain dataset')
        if 'name' not in task.info:
            raise KeyError('task.info must contain name')
        task.info['name'] += f" {datetime.datetime.now().time().isoformat()}"
        task.name = task.info['name']
        dataset_path = self._download_dataset(task.info['dataset'])
        data = DataBundle.load(dataset_path)
        model_folder = Path.home() / 'models' / f'{task.name}'
        env = FileCacheTrainingEnvironment(model_folder)
        success = self._try_run(task, data, env)
        if not success:
            return

        tar_file_name = f'{task.name}.tar.gz'
        tar_dir = Loc.temp_path / 'temp_tar'
        make_tarfile(tar_dir, tar_file_name, model_folder)
        s3_model_path = f'datasphere/{self.project_name}/output/{task.name}/output/model.tar.gz'
        S3YandexHandler.upload_file(self.bucket,
                                    s3_model_path,
                                    tar_dir/tar_file_name)
        print(f'Model uploaded at {s3_model_path}')

    def _download_dataset(self, dataset_name: str) -> Path:
        s3path = f'datasphere/{self.project_name}/datasets/{dataset_name}'
        local_folder = Path.home()/'datasets' / dataset_name
        S3YandexHandler.download_folder(self.bucket, s3path, local_folder)
        return local_folder


def make_tarfile(output_dir, tar_file_name, source_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with tarfile.open(output_dir/tar_file_name, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        print(f'Temp tar is {output_dir/tar_file_name}')

