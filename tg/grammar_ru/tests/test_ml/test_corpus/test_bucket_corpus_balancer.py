from tg.grammar_ru.ml.corpus import CorpusWriter, BucketCorpusBalancer, CorpusReader
from tg.grammar_ru.common import Loc, Separator
from unittest import TestCase


FOLDER = Loc.temp_path/'tests/balancer'

def create_corpora():
    for letter in ['Й','Ц','У']:
        fragment = ''.join([' '.join([letter]+[letter.lower()*10]*cnt) + '. ' for cnt in [3,5,7]])
        fragment = fragment*100
        path = FOLDER/(letter+'.zip')
        writer = CorpusWriter(path, True)
        for i in range(3):
            writer.add_fragment(Separator.separate_string(fragment))
        writer.finalize()


class CorpusBalancerTestCase(TestCase):
    def test_bucket_computation(self):
        create_corpora()
        print(CorpusReader)
        buckets = BucketCorpusBalancer.build_buckets_frame(
            [
                FOLDER / 'Й.zip',
                FOLDER / 'Ц.zip',
                FOLDER / 'У.zip',
            ]
        )
        print(buckets)