from ..locs import Loc
import datetime
from uuid import uuid4
from tg.common.delivery.training.architecture import FileCacheTrainingEnvironment, ResultPickleReader

def _create_id(prefix):
    dt = datetime.datetime.now()
    uid = str(uuid4()).replace('-', '')
    id = f'{prefix}_{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}_{uid}'
    return id

class LocalTrainingRoutine:
    def __init__(self):
        self.training_result_location = Loc.data_path/'training'
        self.bundles_location = Loc.bundles_path

    def execute(self, task, dataset_version: str, wait=True) -> str:
        id = _create_id(task.info.get('name',''))
        folder = self.training_result_location/id
        environment = FileCacheTrainingEnvironment(print,folder)
        task.info['run_at_dataset'] = dataset_version
        task.run_with_environment(self.bundles_location/dataset_version, environment)
        return id

    def get_result(self, id: str):
        return ResultPickleReader(self.training_result_location/id, False)