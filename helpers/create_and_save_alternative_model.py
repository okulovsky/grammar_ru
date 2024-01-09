from tg.common import DataBundle, Loc
from tg.common.ml.batched_training import context as btc
from tg.common.ml.batched_training import sandbox as bts
import pickle


def train_tsa_model():
    tsa_bundle_path = Loc.temp_path /'demos/bundle/bundle'
    tsa_bundle = DataBundle.load(tsa_bundle_path)
    task = bts.AlternativeTrainingTask2()
    task.settings.epoch_count = 20
    task.settings.batch_size = 20000
    task.settings.mini_epoch_count = 5
    task.optimizer_ctor.type = 'torch.optim:Adam'
    task.assembly_point.network_factory.network_type = btc.Dim3NetworkType.AlonAttention
    result = task.run(tsa_bundle)
    model = result['output']['training_task']
    with open(Loc.root_path / 'model/alternative_task_tsa.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_chtoby_model():
    chtoby_bundle_path = Loc.temp_path /'demos/bundle/chtoby_bundle'
    chtoby_bundle = DataBundle.load(chtoby_bundle_path)
    task = bts.AlternativeTrainingTask2()
    task.settings.epoch_count = 20
    task.settings.batch_size = 20000
    task.settings.mini_epoch_count = 5
    task.optimizer_ctor.type = 'torch.optim:Adam'
    task.assembly_point.network_factory.network_type = btc.Dim3NetworkType.AlonAttention
    result = task.run(chtoby_bundle)
    model = result['output']['training_task']
    with open(Loc.root_path / 'model/alternative_task_chtoby.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_ne_model():
    ne_path = Loc.temp_path /'demos/bundle/ne_bundle'
    ne_bundle = DataBundle.load(ne_path)
    task = bts.AlternativeTrainingTask2()
    task.settings.epoch_count = 20
    task.settings.batch_size = 20000
    task.settings.mini_epoch_count = 5
    task.optimizer_ctor.type = 'torch.optim:Adam'
    task.assembly_point.network_factory.network_type = btc.Dim3NetworkType.AlonAttention
    result = task.run(ne_bundle)
    model = result['output']['training_task']
    with open(Loc.root_path / 'model/alternative_task_ne.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    train_tsa_model()
    train_chtoby_model()
    train_ne_model()


if __name__ == '__main__':
    main()
