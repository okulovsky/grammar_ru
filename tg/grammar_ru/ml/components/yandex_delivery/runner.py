from tg.common.delivery.packaging import FakeContainerHandler
from tg.common.delivery.jobs import SSHDockerJobRoutine, DockerOptions
from tg.grammar_ru.ml.components.yandex_delivery.training_example import ClassificationTask, TrainingJob, Loc
from tg.common._common.data_bundle import DataBundle

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
    options=DockerOptions(propagate_environmental_variables=[])
)

routine.local.execute()
# from tg.common.delivery.jobs.ssh_docker_job_routine import build_container

# build_container(job, 'test_job_iris', '1', 'test_iris_img',
#                 image_tag='test_iris_tag')
