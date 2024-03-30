from ....common.delivery.delivery import DependencyList, Packaging, Containering
from ....common.delivery.ssh_docker import SSHDockerConfig, SSHDockerOptions, SSHLocalExecutor, SSHAttachedExecutor, SSHRemoteExecutor
from ....common import Loc
from yo_fluq_ds import FileIO
from pathlib import Path
import os
from pprint import pprint


class DeliveryRoutine:
    def __init__(self, job, name, version):
        self.job = job
        # name, version = Packaging.get_job_name_and_version(job)
        packaging = Packaging(name, version, dict(job=job))

        dependencies = FileIO.read_json(Loc.root_path/'dependencies.json')
        dependencies = DependencyList('defaults', dependencies)
        packaging.dependencies = [dependencies]
        containering = Containering.from_packaging(packaging)
        containering.dependencies = [dependencies]
        self.config = SSHDockerConfig(
            packaging,
            containering,
            SSHDockerOptions(
                env_vatiables_to_propagate=[
                    'AWS_ACCESS_KEY_ID',
                    'AWS_SECRET_ACCESS_KEY'
                ]),
            username=None,
            host=None
        )

    def build_container(self):
        self.config.packaging.make_package()
        self.config.containering.make_container(self.config.packaging)

    # def attached(self):
    #     return SSHAttachedExecutor(self.config)

    def local(self):
        return SSHLocalExecutor(self.config)

    # def remote(self):
    #     return SSHRemoteExecutor(self.config)
