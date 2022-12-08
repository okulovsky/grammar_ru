from typing import *

import boto3
import shutil
import os

from pathlib import Path
from yo_fluq_ds import Query, fluq


class S3YandexHandler:
    @staticmethod
    def download_file(bucket: str, s3_path: str, filename: Path):
        if s3_path.startswith('/'):
            s3_path = s3_path[1:]
        kwargs = S3YandexHandler._get_access_keys()
        client = boto3.client('s3', **kwargs)
        os.makedirs(str(filename.parent), exist_ok=True)
        filename = filename.__str__()
        client.download_file(bucket, s3_path, filename)
        return filename

    @staticmethod
    def download_folder(bucket: str, s3_path: str, folder: Path, report=None):
        if os.path.exists(folder.__str__()):
            shutil.rmtree(folder.__str__())
        os.makedirs(folder.__str__())
        kwargs = S3YandexHandler._get_access_keys()
        s3_resource = boto3.resource('s3', **kwargs)
        bucket_obj = s3_resource.Bucket(bucket)

        keys = [z.key for z in bucket_obj.objects.filter(Prefix=s3_path)]
        keys = Query.en(keys)
        if report == 'tqdm':
            keys = keys.feed(fluq.with_progress_bar())
        client = boto3.client('s3', **kwargs)
        for key in keys:
            proper_key = key[len(s3_path):]
            if proper_key.startswith('/'):
                proper_key = proper_key[1:]
            filename = folder.joinpath(proper_key)
            os.makedirs(str(filename.parent), exist_ok=True)
            filename = filename.__str__()
            client.download_file(bucket, key, filename)

    @staticmethod
    def _get_access_keys():
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
        aws_secret_access = os.environ.get('AWS_SECRET_ACCESS_KEY', None)
        if aws_access_key_id is not None and aws_secret_access is not None:
            kwargs = dict(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access
            )
            kwargs["endpoint_url"] = 'https://storage.yandexcloud.net'
        else:
            raise ValueError(
                f"AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY shouldn't be None")
        return kwargs

    @staticmethod
    def upload_file(bucket_name: str, s3_path: str, filename: Union[Path, str]):
        kwargs = S3YandexHandler._get_access_keys()
        client = boto3.client('s3', **kwargs)
        client.upload_file(str(filename), bucket_name, s3_path)

    @staticmethod
    def upload_folder(bucket_name: str, s3_path: str, folder: Path):
        kwargs = S3YandexHandler._get_access_keys()

        s3 = boto3.resource('s3', **kwargs)
        bucket = s3.Bucket(bucket_name)
        bucket.objects.filter(Prefix=s3_path).delete()

        client = boto3.client('s3', **kwargs)
        for file_path in Query.folder(folder, '**/*'):
            if file_path.is_file():
                file_path_str = file_path.__str__()
                relative_path = file_path.relative_to(folder)
                joint_path = f"{s3_path}/{str(relative_path.as_posix())}"
                client.upload_file(file_path_str, bucket_name, joint_path)

    @staticmethod
    def create_bucket(bucket_name):
        kwargs = S3YandexHandler._get_access_keys()
        client = boto3.client('s3', **kwargs)
        client.create_bucket(Bucket=bucket_name)

    @staticmethod
    def save_to_file(bucket_name: str, s3_path: str, content: str):
        kwargs = S3YandexHandler._get_access_keys()
        client = boto3.client('s3', **kwargs)
        client.put_object(Bucket=bucket_name, Key=s3_path,
                          Body=content, StorageClass='COLD')
