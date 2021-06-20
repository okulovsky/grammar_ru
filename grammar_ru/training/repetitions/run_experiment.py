
from grammar_ru.training.repetitions import repetitions_training_pipeline as stage
from tg.common.ml.batched_training import TrainingSettings, DataBundle
from grammar_ru.training.amenities import Loc, LocalTrainingRoutine


folder = Loc.data_path/'training_result'

if __name__ == '__main__':
    task = stage.Experiment(
        TrainingSettings(50),
        stage.ModelSettings(100000, [10], 10),
    )
    routine = LocalTrainingRoutine()
    routine.execute(task,'repetitions/bundle')
