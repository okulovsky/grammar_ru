from tg.grammar_ru.ml.tasks.tsatsa import TsaTsaTask, NlpAlgorithm
from tg.common.delivery.training import download_and_open_sagemaker_result
from unittest import TestCase
from yo_fluq_ds import FileIO
from tg.common import Loc

class TsaTsaAlgorithmTestCase(TestCase):
    def test_build_unpack(self):
        job_id = 'TSA-PCL7-FALPMSTF-2022-05-26-15-15-08-672'
        alg = TsaTsaTask.Algorithm.build(job_id)
        FileIO.write_pickle(alg, Loc.temp_path/'tests/tsatsa.pickle')

        alg = FileIO.read_pickle(Loc.temp_path/'tests/tsatsa.pickle') #type: TsaTsaTask.Algorithm
        df = alg.run_on_string('Мне нравиться, что вы больны не мой. Мне нравится, что я больна не вами. Я хочу нравится.')
        print(df)
        self.assertEqual(2, df.shape[0])
        self.assertListEqual([1,20], list(df.word_id))
        self.assertTrue((df.error_type=='grammatic').all())
        self.assertEqual(['нравится','нравиться'], list(df.suggest))
