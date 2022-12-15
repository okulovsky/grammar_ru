import datetime
from dotenv import load_dotenv
from tg.grammar_ru.common import Loc
from tg.grammar_ru.ml.components.yandex_delivery.training_job import TrainingJob
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.common.delivery.packaging import FakeContainerHandler
from tg.grammar_ru.ml.components.yandex_delivery.docker_tools import deploy_container
from tg.common.delivery.jobs.ssh_docker_job_routine import build_container
from tg.grammar_ru.ml.components.training_task_factory import TaskFactory
from tg.grammar_ru.ml.tasks.antecedents.task_factory import AntecedentCandidateTask

project_name = 'antecedents_project'
dataset_name = 'antecedents_simple_dataset'
bucket = 'antecedents-simple'

tag = 'v_' + datetime.datetime.now().time().strftime("%H_%M_%S")
dockerhub_repo = 'grammar_repo'
dockerhub_login = 'woiperdinger'
local_img = 'antc_img'


class TestTask(TaskFactory):
    def create_task(self, data, env):
        print('task created')


def get_training_job() -> TrainingJob:
    #task = TestTask()
    task = AntecedentCandidateTask()
    task.info["dataset"] = dataset_name
    task.info["name"] = "antc_task"

    job = TrainingJob(
        tasks=[task],
        project_name=project_name,
        bucket=bucket
    )
    return job


def local_job_execution(local_job):
    routine = SSHDockerJobRoutine(
        job=local_job,
        remote_host_address=None,
        remote_host_user=None,
        handler_factory=FakeContainerHandler.Factory(),
        options=DockerOptions(propagate_environmental_variables=["AWS_ACCESS_KEY_ID",
                                                                 "AWS_SECRET_ACCESS_KEY"])
    )
    routine.local.execute()


def build_deploy(job):
    build_container(job, 'antc_job', '1', local_img, image_tag=tag)
    deploy_container(local_img, dockerhub_repo, dockerhub_login, tag)


if __name__ == "__main__":
    load_dotenv(Loc.root_path / 'environment.env')
    print('a')
    job = get_training_job()
    print('b')
    build_deploy(job)
    print('c')
