from typing import List
from pathlib import Path
import datetime
import tarfile
import os.path
import shutil
from .....common import DataBundle
from .....common.delivery.training.architecture import FileCacheTrainingEnvironment
from .....common import Logger
from .....common.delivery.jobs import DeliverableJob
from ....common.loc import Loc
from ...components.training_task_factory import TaskFactory
from ..yandex_storage.s3_yandex_helpers import S3YandexHandler
import sys


class TrainingJob(DeliverableJob):
    def __init__(self, tasks: List[TaskFactory], project_name: str, bucket: str):
        super().__init__()
        self.tasks = tasks
        self.project_name = project_name
        self.bucket = bucket

    def get_name_and_version(self):
        return 'datasphere_job', 'v1'

    def run(self):
        if 'AWS_ACCESS_KEY_ID' not in os.environ:
            os.environ['AWS_ACCESS_KEY_ID'] = sys.argv[1]
            os.environ['AWS_SECRET_ACCESS_KEY'] = sys.argv[2]

        self.name = f"job_{self.project_name}_{datetime.datetime.now().time().isoformat()}"
        Logger.info("Running list of tasks")
        for task in self.tasks:
            self._run_task(task)
        tasks_list_s3_path = self._upload_tasks_list()
        Logger.info(f"List of tasks uploaded to {tasks_list_s3_path}")
        Logger.info(f"Training job {self.name} is done")

    def _try_run(self, task, data, env):
        try:
            Logger.info(f"Running task {task.name} with environment")
            task.run_with_environment(data, env)
            return True
        except Exception as e:
            S3YandexHandler.save_to_file(self.bucket,
                                         s3_path=f"datasphere/{self.project_name}/exceptions/{task.name}.txt",
                                         content=str(e))
            Logger.info(f'Exception in {task.name} loaded to DataSphere')
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
        Logger.info("Creating FileCacheTrainingEnvironment")
        env = FileCacheTrainingEnvironment(model_folder)
        success = self._try_run(task, data, env)
        if not success:
            return

        tar_file_name = f'{task.name}.tar.gz'
        tar_dir = Loc.temp_path / 'temp_tar'
        Logger.info(f"Creating tar file from task's output")
        make_tarfile(tar_dir, tar_file_name, model_folder)
        s3_model_path = f'datasphere/{self.project_name}/output/{task.name}/output/model.tar.gz'
        Logger.info("Uploading model output to DataSphere")
        S3YandexHandler.upload_file(self.bucket,
                                    s3_model_path,
                                    tar_dir / tar_file_name)
        print(f'Model uploaded at {s3_model_path}')

    def _upload_tasks_list(self) -> str:
        s3_path = f'datasphere/{self.project_name}/job_info/{self.name}.txt'
        S3YandexHandler.save_to_file(self.bucket,
                                     s3_path=s3_path,
                                     content=str([task.name for task in self.tasks]))
        return s3_path

    def _download_dataset(self, dataset_name: str) -> Path:
        Logger.info("Downloading dataset")
        s3path = f'datasphere/{self.project_name}/datasets/{dataset_name}'
        local_folder = Path.home() / 'datasets' / dataset_name
        S3YandexHandler.download_folder(self.bucket, s3path, local_folder)
        return local_folder


def make_tarfile(output_dir, tar_file_name, source_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with tarfile.open(output_dir / tar_file_name, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
