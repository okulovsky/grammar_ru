from tg.common.delivery.jobs.ssh_docker_job_routine import build_container
import subprocess
from tg.common.delivery.packaging import FakeContainerHandler
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.grammar_ru.ml.components.yandex_delivery.training_example import ClassificationTask, TrainingJob, Loc
from tg.common._common.data_bundle import DataBundle
from tg.grammar_ru.ml.components.yandex_delivery.docker_tools import deploy_container
from dotenv import load_dotenv
import datetime
load_dotenv(Loc.root_path / 'environment.env')

project_name = 'testirisproject'
dataset_name = 'irisdataset'
bucket = 'testirisbucket'


task = ClassificationTask()
task.info['dataset'] = dataset_name
task.info['name'] = 'classification_iris_task'

bundle = DataBundle.load(Loc.temp_path / 'temp_bundle')
# task.run(bundle)

job = TrainingJob(tasks=[task],
                  project_name=project_name,
                  bucket=bucket)


routine = SSHDockerJobRoutine(
    job=job,
    remote_host_address=None,
    remote_host_user=None,
    handler_factory=FakeContainerHandler.Factory(),
    options=DockerOptions(propagate_environmental_variables=["AWS_ACCESS_KEY_ID",
                                                             "AWS_SECRET_ACCESS_KEY"])
)
tag = 'v_' + datetime.datetime.now().time().strftime("%H_%M_%S")
dockerhub_repo = 'grammar_repo'
local_img = 'test_iris_img'

dockerhub_login = 'sergio0x0'
# 6a
# routine.local.execute()

# 6b

build_container(job, 'test_job_iris', '1', local_img,
                image_tag=tag)


deploy_container(local_img, dockerhub_repo, dockerhub_login, tag)
