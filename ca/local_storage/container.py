import os
import pickle
import copy
from typing import *
from dataclasses import dataclass
from pathlib import Path
import shutil
from yo_fluq_ds import FileIO,Query
from grammar_ru.common import Loc
import datetime
import subprocess
from tg.common.delivery.delivery import ContainerPusher



@dataclass
class Container:
    name: str
    tag: str
    entry_point: Any
    dependencies: List[str]
    deployed_folders: Iterable[str]
    pusher: Optional[ContainerPusher]
    python_version: str = '3.8'
    root_path: Optional[Path] = None
    additional_dependencies: Optional[Iterable[Iterable[str]]] = None
    custom_dockerfile_template: str = None
    custom_entryfile_template: str = None
    custom_setup_py_template: str = None
    custom_dockerignore_template: str = None


    def _make_pip_install(self, deps: Iterable[str]):
        deps = list(deps)
        if len(deps) == 0:
            return ''
        libs = ' '.join(deps)
        return 'RUN pip install ' + libs

    def _make_dockerfile(self):
        dockerfile_template = DOCKERFILE_TEMPLATE
        if self.custom_dockerfile_template is not None:
            dockerfile_template = self.custom_dockerfile_template

        folders = list(self.deployed_folders)

        install_libraries = self._make_pip_install(self.dependencies)
        if self.additional_dependencies is not None:
            for dep_list in self.additional_dependencies:
                install_libraries+='\n\n'+self._make_pip_install(dep_list)

        content = dockerfile_template.format(
            python_version=self.python_version,
            install_libraries=install_libraries,
        )
        return content

    def _make_entryfile(self):
        entryfile_template = ENTRYFILE_TEMPLATE
        if self.custom_entryfile_template is not None:
            entryfile_template = self.custom_entryfile_template

        return entryfile_template.format(name=self.name, version=self.tag)

    def _make_setup_py(self):
        template = SETUP_PY_TEMPLATE
        if self.custom_setup_py_template is not None:
            template = self.custom_setup_py_template
        return template.format(name=self.name, version = self.tag)


    def _make_dockerignore(self):
        template = DOCKER_IGNORE_TEMPLATE
        if self.custom_dockerignore_template is not None:
            template = self.custom_dockerignore_template
        return template


    def _make_deploy_folder(self, root_path: Path, docker_path: Path):
        shutil.rmtree(docker_path, ignore_errors=True)
        os.makedirs(docker_path)
        FileIO.write_text(self._make_dockerfile(), docker_path/'Dockerfile')
        FileIO.write_text(self._make_setup_py(), docker_path/'setup.py')
        FileIO.write_text(self._make_dockerignore(), docker_path/'.dockerignore')
        FileIO.write_text(self._make_entryfile(), docker_path/'entry.py')
        for folder in self.deployed_folders:
            shutil.copytree(root_path/folder, docker_path/folder)


    def build(self):
        root_path = self.root_path
        if root_path is None:
            root_path = Loc.root_path
        docker_path = root_path/'deployments'/str(datetime.datetime.now().timestamp())
        self._make_deploy_folder(root_path, docker_path)
        with open(docker_path / 'entry.pkl', 'wb') as file:
            pickle.dump(self.entry_point, file)
        arguments = ['docker', 'build', str(docker_path), '--tag', f'{self._docker_conform_name(self.name)}:{self.tag}']

        print(str(docker_path))

        print(f'{self._docker_conform_name(self.name)}:{self.tag}')
        if subprocess.call(arguments) != 0:
            raise ValueError(f'Docker call caused an error. Arguments\n{" ".join(arguments)}')
        shutil.rmtree(docker_path)

    def _docker_conform_name(self, name: str):
        return name.lower()

    def push(self):
        if self.pusher is None:
            raise ValueError('Push is requested, but no pusher is set')
        self.pusher.push(self._docker_conform_name(self.name), self.tag)

    def get_remote_name(self):
        if self.pusher is None:
            raise ValueError('get_remote_name is requested, but no pusher is set')
        return self.pusher.get_remote_name(self._docker_conform_name(self.name), self.tag)








ENTRYFILE_TEMPLATE = '''
from tg.common.delivery.delivery import EntryPoint
from pathlib import Path

entry_point = EntryPoint("{name}", "{version}", Path(__file__).parent/"entry.pkl")
entry_point.run()

'''


DOCKERFILE_TEMPLATE = '''FROM python:{python_version}

{install_libraries}

COPY . /

RUN pip install -e .

CMD ["python3","/entry.py"]
'''

DOCKER_IGNORE_TEMPLATE = '''
**/*.pyc
'''

SETUP_PY_TEMPLATE = '''
from setuptools import setup, find_packages

setup(name='{name}',
      version='{version}',
      packages=find_packages(),
      install_requires=[
      ],
      include_package_data = True,
      zip_safe=False)
'''


class SimpleContainerPusher(ContainerPusher):
    def __init__(self, docker_url, registry, login, password):
        self.registry = registry
        self.docker_url = docker_url
        self.login = login
        self.password = password

    def _try_execute(self, arguments):
        if subprocess.call(list(arguments)) != 0:
            raise ValueError(f'Error when running command\n{" ".join(arguments)}')


    def get_auth_command(self) -> Iterable[str]:
        return ['docker', 'login', self.docker_url, '--username', self.login, '--password',
                self.password]


    def push(self, image_name: str, image_tag: str):
        self._try_execute(['docker','tag',f'{self.registry}:{image_tag}',f'{self.docker_url}/{self.registry}:latest'])
        self._try_execute(self.get_auth_command())
        self._try_execute(['docker','push',f'{self.docker_url}/{self.registry}:latest'])


class Deployment:
    container: Container
    ssh_url: str
    ssh_username: str
    open_ports: Iterable[int] = (7860,)
    mount_remote_data_folder: Optional[str] = None
    propagate_env_variables: Iterable[str] = ()
    custom_env_variables: Dict[str, str] = None
    additional_arguments: Iterable[str] = ()


    @property
    def pusher(self) -> SimpleContainerPusher:
        return self.container.pusher

    def _ssh(self):
        return [
            'ssh',
            f'{self.ssh_username}@{self.ssh_url}'
        ]

    def _remote_pull(self):
        subprocess.call(self._ssh() + list(self.pusher.get_auth_command()))
        subprocess.call(self._ssh() + ['docker', 'pull', f'{self.pusher.docker_url}/{self.pusher.registry}:latest'])

    def _get_run_local(self, image):
        ports = Query.en(self.open_ports).select_many(lambda z: ['-p', f'{z}:{z}']).to_list()
        args = ['docker', 'run'] + ports + [image]
        return args

    def kill_remote(self):
        reply = subprocess.check_output(self._ssh() +
                                        ['docker',
                                         'ps',
                                         '-q',
                                         '--filter',
                                         f'ancestor={self.pusher.docker_url}/{self.pusher.registry}',
                                         ])
        container = reply.decode('utf-8').strip()
        if container != '':
            subprocess.call(self._ssh() + ['docker', 'stop', container])

    def _get_run(self, image):
        ports = Query.en(self.open_ports).select_many(lambda z: ['-p', f'{z}:{z}']).to_list()
        mount = []
        if self.mount_remote_data_folder is not None:
            mount = ['--mount', f'type=bind,source={self.mount_remote_data_folder},target=/data']

        if self.custom_env_variables is None:
            env_dict = {}
        else:
            env_dict = copy.deepcopy(self.custom_env_variables)
        for var in self.propagate_env_variables:
            env_dict[var] = os.environ[var]

        envs = []
        for var, value in env_dict.items():
            environment_quotation = '"'
            envs.append('--env')
            envs.append(f'{environment_quotation}{var}={value}{environment_quotation}')

        args = ['docker', 'run'] + ports + mount + envs + list(self.additional_arguments) + [image]
        return args

    def run_remote(self):
        self.container.build()
        self.container.push()
        self.kill_remote()
        self._remote_pull()
        if self.mount_remote_data_folder is not None:
            subprocess.call(self._ssh() + ['mkdir', '-p', self.mount_remote_data_folder])
        commands = self._ssh() + self._get_run(f'{self.pusher.docker_url}/{self.pusher.registry}:latest')
        subprocess.call(commands)


class ExampleJob:
    def __init__(self, check_environmental_variables = None):
        if check_environmental_variables is None:
            self.check_environmental_variables = []
        else:
            self.check_environmental_variables = list(check_environmental_variables)


    def run(self):
        for env in self.check_environmental_variables:
            Logger.info(f'Variable {env} is found: {env in os.environ}')
        Logger.info('SUCCESS')