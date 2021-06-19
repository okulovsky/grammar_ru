
from grammar_ru.training.repetitions import repetitions_training_pipeline as stage
from tg.common.ml.batched_training import TrainingSettings, DataBundle
from grammar_ru.training.amenities import Loc
from tg.common.delivery.training.architecture import FileCacheTrainingEnvironment


folder = Loc.data_path/'training_result'

if __name__ == '__main__':
    task = stage.Experiment(
        TrainingSettings(10),
        stage.ModelSettings(200000, [20], 10),
    )
    env = FileCacheTrainingEnvironment(print, folder)
    task.run_with_environment(DataBundle.load(Loc.bundles_path/'repetitions/test_bundle'), env)
