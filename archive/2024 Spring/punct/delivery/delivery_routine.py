from yo_fluq_ds import FileIO

from ....common.delivery.delivery import DependencyList, Packaging, Containering
from ....common.delivery.ssh_docker import SSHDockerConfig, SSHDockerOptions, SSHLocalExecutor
from ....common import Loc


class DeliveryRoutine:
    def __init__(self, job, name, version):
        self.job = job
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

    def local(self):
        return SSHLocalExecutor(self.config)
