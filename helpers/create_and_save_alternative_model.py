import torch

from tg.common import DataBundle, Loc
from tg.grammar_ru import Separator
from tg.grammar_ru.algorithms import alternative
from tg.grammar_ru.features import SnowballFeaturizer
from tg.common.ml.batched_training import context as btc
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import sandbox as bts
import pickle



def main():
    tsa_bundle_path = Loc.temp_path / 'demos/bundle/bundle'
    tsa_bundle = DataBundle.load(tsa_bundle_path)
    task = bts.AlternativeTrainingTask2()
    task.settings.epoch_count = 1
    task.settings.batch_size = 20000
    task.settings.mini_epoch_count = 5
    task.optimizer_ctor.type = 'torch.optim:Adam'
    task.assembly_point.network_factory.network_type = btc.Dim3NetworkType.AlonAttention
    result = task.run(tsa_bundle)
    model = result['output']['training_task']
    with open (Loc.root_path / 'model/alternative_task.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('ok')

if __name__ == '__main__':
    main()
