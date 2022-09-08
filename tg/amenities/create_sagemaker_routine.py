import os
import subprocess

from pathlib import Path

from tg.common._common import Loc
from tg.common.delivery.training import SagemakerTrainingRoutine
from tg.common.delivery.packaging import ContainerHandler



class ECRHandler(ContainerHandler):
    """
    This pushes the image to the ECT repository.
    To avoid confusion: in our setup, we have one repository where we push EVERYTHING related to training grounds.
    Why? Because it has a lot of dependencies, and if we use the new repository for each project, we will have to
    upload lots of data each time. So, we use one registry and incorporate a name of the project in the tag.
    """

    def __init__(self,
                 name: str,
                 version: str,
                 region: str = 'eu-west-1',
                 aws_id: str = '148702677388',
                 repo_name: str = 'ps-data'):
        self.name = name
        self.verion = version
        self.region = region
        self.aws_id = aws_id
        self.registry = f'{self.aws_id}.dkr.ecr.{self.region}.amazonaws.com'
        self.image_name = repo_name
        self.image_tag = f'{name}-{version}'

    def get_image_name(self) -> str:
        return self.image_name

    def get_tag(self) -> str:
        return self.image_tag

    def get_remote_name(self):
        return f'{self.registry}/{self.image_name}:{self.image_tag}'

    def get_auth_command(self):
        return ['aws', 'ecr', 'get-login-password', '--region', self.region, '|',
                'docker', 'login', '--username', 'AWS', '--password-stdin',
                f'{self.aws_id}.dkr.ecr.{self.region}.amazonaws.com']

    def push(self):
        """
        Pushes the previously build local docker container to AWS ECR, from where it can be later used by Sagemaker
        """
        p1 = subprocess.Popen(['aws', 'ecr', 'get-login-password', '--region', self.region], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['docker', 'login', '-u', 'AWS', '--password-stdin', self.registry], stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()
        p2.communicate()

        remote_name = self.get_remote_name()

        subprocess.call([
            'docker',
            'tag',
            self.image_name + ":" + self.image_tag,
            remote_name
        ])

        subprocess.call([
            'docker',
            'push',
            remote_name
        ])

    class Factory(ContainerHandler.Factory):
        def create_handler(self, name: str, version: str) -> 'ContainerHandler':
            return ECRHandler(name, version)



def create_sagemaker_routine(
        project_name: str,
        instance_type: str = 'ml.m4.xlarge',
        local_dataset_storage=None,
        s3_bucket='ps-data-science-sandbox',
        prefix='sagemaker'
):
    local_dataset_storage = local_dataset_storage or Loc.data_cache_path / 'bundles' / project_name
    region = 'eu-west-1'
    aws_id = os.environ.get('AWS_ID')
    repo_name = 'ps-data'
    registry = f'{aws_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}'
    return SagemakerTrainingRoutine(
        local_dataset_storage,
        project_name,
        ECRHandler.Factory(),
        os.environ['AWS_ROLE'],
        'ps-data-science-sandbox'
    )
