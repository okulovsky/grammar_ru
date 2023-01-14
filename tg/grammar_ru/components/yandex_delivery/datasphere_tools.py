from ....common._common import Loc
from ....common.delivery.training.architecture import ResultPickleReader
import os
import boto3
import sagemaker
import shutil
import subprocess
from ..yandex_storage.s3_yandex_helpers import S3YandexHandler
from pathlib import Path
import ast 
from typing import List

_TRAINING_RESULTS_LOCATION = Loc.temp_path / 'training_results'


def open_datasphere_result(filename, job_id):
    folder = _TRAINING_RESULTS_LOCATION / (job_id + '.unzipped')
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    tar_call = ['tar', '-C', folder, '-xvf', filename]
    subprocess.call(tar_call)
    return ResultPickleReader(Path(folder))


def download_and_open_datasphere_result(bucket, project_name, task_id, dont_redownload=False):
    filename = _TRAINING_RESULTS_LOCATION / f'{task_id}.tar.gz'
    folder = _TRAINING_RESULTS_LOCATION / task_id
    if filename.is_file() and folder.is_dir() and dont_redownload:
        return ResultPickleReader(Path(folder))
    else:
        path = f'datasphere/{project_name}/output/{task_id}/output/model.tar.gz'
        S3YandexHandler.download_file(
            bucket,
            path,
            filename
        )
        return open_datasphere_result(filename, task_id)


def get_tasks_list(list_s3_path:str, bucket)->List[str]:
    tmp_local_file = Loc.temp_path / list_s3_path.split('/')[-1]
    S3YandexHandler.download_file(bucket, list_s3_path, tmp_local_file)
    with open(tmp_local_file,'r') as f:
        tasks = ast.literal_eval(f.read())
    return tasks