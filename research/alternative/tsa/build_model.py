from tg.common.delivery.training import download_and_open_sagemaker_result
from yo_fluq_ds import *
from tg.common.ml.miscellaneous import roc_optimal_threshold
from pathlib import Path
from tg.grammar_ru.ml.tasks.tsa import TsaAlgorithm
from tg.grammar_ru.algorithms import NlpAlgorithm
from tg.grammar_ru import Separator, Loc
import os

path = Loc.models_path / 'tsa.pickle'

def build_algorithm(job_id, only_pymorphy):
    result = download_and_open_sagemaker_result('ps-data-science-sandbox', 'tsa', job_id, True)
    rdf = pd.read_parquet(result.get_path('output/result_df.parquet'))
    borderline = roc_optimal_threshold(rdf.loc[rdf.stage == 'display'].true,
                                       rdf.loc[rdf.stage == 'display'].predicted)
    model = result.unpickle('output/training_task.pkl')
    words = FileIO.read_json(Path(__file__).parent/'words.json')
    return TsaAlgorithm(model, words, borderline, only_pymorphy)

def store_algorithm(job_id, only_pymorphy):
    alg = build_algorithm(job_id, only_pymorphy)
    os.makedirs(path.parent, exist_ok=True)
    FileIO.write_pickle(alg, path)
    print(path.stat().st_size)


def test_algorithm(alg: TsaAlgorithm):
    text = "Он привык всем нравиться. Ему нравиться, когда им восхищаются"
    db = Separator.build_bundle(text)
    rdf = alg.run(db, db.src.index)
    t = rdf.iloc[6]
    assert t['error'] == True
    assert t['algorithm'] == 'tsa'
    assert t['error_type'] == NlpAlgorithm.ErrorTypes.Grammatic
    assert t['suggest'] == 'нравится'


if __name__ == '__main__':
    pass
    store_algorithm('TSAG-PCL15-2022-08-20-18-55-01-132', False)
    alg = FileIO.read_pickle(path)
    test_algorithm(alg)

