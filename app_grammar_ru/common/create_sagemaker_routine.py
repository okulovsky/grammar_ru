from typing import *
from ..common._common import Loc
from ..common.delivery.delivery import ContainerPusher
import subprocess
from ..common.delivery.delivery import Packaging, Containering, DependencyList
from ..common.delivery.sagemaker import SagemakerOptions, SagemakerConfig, SagemakerJob, SagemakerAttachedExecutor, SagemakerLocalExecutor, SagemakerRemoteExecutor, DOCKERFILE_TEMPLATE
from ..common.ml.training_core import AbstractTrainingTask
from yo_fluq_ds import FileIO
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent/'environment.env')

class DoctoredECRPusher(ContainerPusher):
    def __init__(self,
                 region: str,
                 aws_id: str,
                 repo_name: str,
                 aws_credentials_profile = None
                 ):
        self.region = region
        self.aws_id = aws_id
        self.repo_name = repo_name
        self.aws_credentials_profile = aws_credentials_profile
        self.registry = f'{self.aws_id}.dkr.ecr.{self.region}.amazonaws.com'

    def get_login_password(self):
        result = ['aws', 'ecr', 'get-login-password', '--region', self.region]
        if self.aws_credentials_profile is not None:
            result.append('--profile')
            result.append(self.aws_credentials_profile)
        return result

    def get_auth_command(self) -> Iterable[str]:
        result = self.get_login_password()
        result += ['|', 'docker', 'login', '--username', 'AWS', '--password-stdin', f'{self.aws_id}.dkr.ecr.{self.region}.amazonaws.com']
        return result

    def get_remote_name(self, image_name: str, image_tag: str) -> str:
        remote_name = f'{self.registry}/{self.repo_name}:{image_name}-{image_tag}'
        return remote_name

    def push(self, image_name: str, image_tag: str):
        """
        Pushes the previously build local docker container to AWS ECR, from where it can be later used by Sagemaker
        """
        p1 = subprocess.Popen(self.get_login_password(), stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['docker', 'login', '-u', 'AWS', '--password-stdin', self.registry], stdin=p1.stdout,
                              stdout=subprocess.PIPE)
        p1.stdout.close()
        p2.communicate()

        remote_name = self.get_remote_name(image_name, image_tag)

        subprocess.call([
            'docker',
            'tag',
            image_name + ":" + image_tag,
            remote_name
        ])

        subprocess.call([
            'docker',
            'push',
            remote_name
        ])




class SagemakerRoutine:
    def __init__(self,
                 task: AbstractTrainingTask,
                 dataset: Optional[str] = None,
                 project_name: Optional[str] = None,
                 dependencies_file = None
                 ):
        name = type(task).__name__
        version = '0'
        if hasattr(task, 'info') and isinstance(task.info, dict):
            name = task.info.get('name', name)
            dataset = task.info.get('dataset', dataset)
            project_name = task.info.get('project_name', project_name)

        if dataset is None:
            raise ValueError('Dataset was not provided')
        if project_name is None:
            raise ValueError('Project name was not provided')

        job = SagemakerJob(task)

        packaging = Packaging(name, version, dict(job=job))
        if dependencies_file is not None:
            dependencies = FileIO.read_json(dependencies_file)
        else:
            dependencies = DEFAULT_DEPENDENCIES

        dependencies = DependencyList('defaults', dependencies)
        packaging.dependencies = [dependencies]

        containering = Containering.from_packaging(packaging)
        containering.dependencies = [dependencies]
        containering.pusher = DoctoredECRPusher(
            os.environ['SAGEMAKER_REGION'],
            os.environ['SAGEMAKER_ID'],
            os.environ['SAGEMAKER_CONTAINER_REGISTRY'],
        )
        containering.dockerfile_template = DOCKERFILE_TEMPLATE
        containering.run_file_name='train.py'

        settings = SagemakerOptions(
            os.environ['AWS_ROLE'],
            os.environ['SAGEMAKER_BUCKET'],
            project_name,
            Loc.data_cache_path/'bundles',
            dataset,
        )

        containering.python_version = '3.8'
        settings.use_spot_instances = True
        #packaging.tg_name = 'tgv3'

        self.config = SagemakerConfig(
            job,
            packaging,
            containering,
            settings
        )

    def attached(self):
        return SagemakerAttachedExecutor(self.config)

    def local(self):
        return SagemakerLocalExecutor(self.config)

    def remote(self):
        return SagemakerRemoteExecutor(self.config)





DEFAULT_DEPENDENCIES = (

['anyio==3.6.2', 'argon2-cffi==21.3.0', 'argon2-cffi-bindings==21.2.0', 'arrow==1.2.3', 'asttokens==2.2.1', 'attrs==22.2.0', 'backcall==0.2.0', 'beautifulsoup4==4.11.1', 'bleach==5.0.1', 'boto3==1.26.54', 'botocore==1.29.54', 'cffi==1.15.1', 'click==8.1.3', 'comm==0.1.2', 'contextlib2==21.6.0', 'contourpy==1.0.7', 'corus==0.9.0', 'coverage==7.0.5', 'cramjam==2.6.2', 'cycler==0.11.0', 'DAWG-Python==0.7.2', 'debugpy==1.6.5', 'decorator==5.1.1', 'defusedxml==0.7.1', 'Deprecated==1.2.13', 'dill==0.3.6', 'docopt==0.6.2', 'entrypoints==0.4', 'executing==1.2.0', 'fastjsonschema==2.16.2', 'fastparquet==2023.1.0', 'Flask==2.1.0', 'fonttools==4.38.0', 'fqdn==1.5.1', 'fsspec==2023.1.0', 'google-pasta==0.2.0', 'idna==3.4', 'importlib-metadata==4.13.0', 'importlib-resources==5.10.2', 'ipykernel==6.20.2', 'ipython==8.8.0', 'ipython-genutils==0.2.0', 'ipywidgets==8.0.4', 'isoduration==20.11.0', 'itsdangerous==2.1.2', 'jedi==0.18.2', 'Jinja2==3.1.2', 'jmespath==1.0.1', 'joblib==1.2.0', 'jsonpickle==3.0.1', 'jsonpointer==2.3', 'jsonschema==4.17.3', 'jupyter==1.0.0', 'jupyter-console==6.4.4', 'jupyter-events==0.6.3', 'jupyter_client==7.4.9', 'jupyter_core==5.1.3', 'jupyter_server==2.1.0', 'jupyter_server_terminals==0.4.4', 'jupyterlab-pygments==0.2.2', 'jupyterlab-widgets==3.0.5', 'kiwisolver==1.4.4', 'MarkupSafe==2.1.2', 'matplotlib==3.6.3', 'matplotlib-inline==0.1.6', 'mistune==2.0.4', 'multiprocess==0.70.14', 'navec==0.10.0', 'nbclassic==0.4.8', 'nbclient==0.7.2', 'nbconvert==7.2.8', 'nbformat==5.7.3', 'nerus==1.7.0', 'nest-asyncio==1.5.6', 'notebook==6.5.2', 'notebook_shim==0.2.2', 'numpy==1.24.1', 'nvidia-cublas-cu11==11.10.3.66', 'nvidia-cuda-nvrtc-cu11==11.7.99', 'nvidia-cuda-runtime-cu11==11.7.99', 'nvidia-cudnn-cu11==8.5.0.96', 'packaging==23.0', 'pandas==1.5.3', 'pandocfilters==1.5.0', 'parso==0.8.3', 'pathos==0.3.0', 'patsy==0.5.3', 'pexpect==4.8.0', 'pickleshare==0.7.5', 'Pillow==9.4.0', 'pkgutil_resolve_name==1.3.10', 'platformdirs==2.6.2', 'pox==0.3.2', 'ppft==1.7.6.6', 'prometheus-client==0.15.0', 'prompt-toolkit==3.0.36', 'protobuf==3.20.3', 'protobuf3-to-dict==0.1.5', 'psutil==5.9.4', 'ptyprocess==0.7.0', 'pure-eval==0.2.2', 'pyaml==21.10.1', 'pyarrow==10.0.1', 'pycparser==2.21', 'pyenchant==3.2.2', 'Pygments==2.14.0', 'pymorphy2==0.9.1', 'pymorphy2-dicts-ru==2.4.417127.4579844', 'pyparsing==3.0.9', 'pyrsistent==0.19.3', 'python-dateutil==2.8.2', 'python-dotenv==0.21.1', 'python-json-logger==2.0.4', 'pytz==2022.7.1', 'PyYAML==6.0', 'pyzmq==25.0.0', 'qtconsole==5.4.0', 'QtPy==2.3.0', 'razdel==0.5.0', 'rfc3339-validator==0.1.4', 'rfc3986-validator==0.1.1', 's3transfer==0.6.0', 'sagemaker==2.129.0', 'schema==0.7.5', 'scikit-learn==1.2.0', 'scipy==1.10.0', 'seaborn==0.12.2', 'Send2Trash==1.8.0', 'simplejson==3.18.1', 'six==1.16.0', 'sklearn==0.0.post1', 'slovnet==0.5.0', 'smdebug-rulesconfig==1.0.1', 'sniffio==1.3.0', 'soupsieve==2.3.2.post1', 'stack-data==0.6.2', 'statsmodels==0.13.5', 'terminado==0.17.1', 'threadpoolctl==3.1.0', 'tinycss2==1.2.1', 'torch==1.13.1', 'tornado==6.2', 'tqdm==4.64.1', 'traitlets==5.8.1', 'typing_extensions==4.4.0', 'uri-template==1.2.0', 'urllib3==1.26.14', 'wcwidth==0.2.6', 'webcolors==1.12', 'webencodings==0.5.1', 'websocket-client==1.4.2', 'Werkzeug==2.2.2', 'widgetsnbextension==4.0.5', 'wrapt==1.14.1', 'yo-fluq==1.1.11', 'yo-fluq-ds==1.1.11', 'zipp==3.11.0']

)