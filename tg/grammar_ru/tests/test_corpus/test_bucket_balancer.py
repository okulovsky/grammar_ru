from tg.grammar_ru.ml.corpus import CorpusWriter, BucketBalancer, CorpusReader
from tg.grammar_ru.common import Loc, Separator
from unittest import TestCase
from yo_fluq_ds import *

FOLDER = Loc.temp_path/'tests/balancer'

def create_corpora():
    corpora = []
    for letter in ['Й','Ц','У']:
        fragment = ''.join([' '.join([letter]+[letter.lower()*10]*cnt) + '. ' for cnt in [3,5,7]])
        fragment = fragment*10
        path = FOLDER/(letter+'.zip')
        writer = CorpusWriter(path, True)
        for i in range(3):
            writer.add_fragment(Separator.separate_string(fragment))
        writer.finalize()
        corpora.append(path)
    return corpora


class CorpusBalancerTestCase(TestCase):
    def test_bucket_computation(self):
        corpora= create_corpora()
        buckets = BucketBalancer.collect_buckets(CorpusReader.read_frames_from_several_corpora(corpora))
        pd.options.display.max_rows=None
        bstat = buckets.groupby(['corpus_id','log_len']).size().to_frame('sz').reset_index()
        bstat = Query.df(bstat).to_list()
        self.assertListEqual(
            [{'corpus_id': 'Й.zip', 'log_len': 2, 'sz': 60}, {'corpus_id': 'Й.zip', 'log_len': 3, 'sz': 30},
             {'corpus_id': 'У.zip', 'log_len': 2, 'sz': 60}, {'corpus_id': 'У.zip', 'log_len': 3, 'sz': 30},
             {'corpus_id': 'Ц.zip', 'log_len': 2, 'sz': 60}, {'corpus_id': 'Ц.zip', 'log_len': 3, 'sz': 30}],
            bstat
        )
        buckets = buckets.feed(fluq.add_ordering_column(['corpus_id','log_len'], 'sentence_id'))
        buckets = buckets.loc[buckets.order<2]
        take = BucketBalancer.buckets_statistics_to_dict(buckets)
        self.assertDictEqual(
            {'Й.zip': {0, 1, 2, 5}, 'У.zip': {0, 1, 2, 5}, 'Ц.zip': {0, 1, 2, 5}},
            take
        )
        balancer = BucketBalancer(take)
        dfs = CorpusReader.read_frames_from_several_corpora(corpora).select(lambda z: balancer.select(None,z,None)).to_list()
        rdf = pd.concat(dfs)
        rstat = rdf.groupby(['corpus_id','sentence_id']).size().to_frame('sz').reset_index().sort_values(['corpus_id','sentence_id'])
        rstat = Query.df(rstat).to_list()
        self.assertListEqual(
            [{'corpus_id': 'Й.zip', 'sentence_id': 0, 'sz': 5}, {'corpus_id': 'Й.zip', 'sentence_id': 1, 'sz': 7},
             {'corpus_id': 'Й.zip', 'sentence_id': 2, 'sz': 9}, {'corpus_id': 'Й.zip', 'sentence_id': 5, 'sz': 9},
             {'corpus_id': 'У.zip', 'sentence_id': 0, 'sz': 5}, {'corpus_id': 'У.zip', 'sentence_id': 1, 'sz': 7},
             {'corpus_id': 'У.zip', 'sentence_id': 2, 'sz': 9}, {'corpus_id': 'У.zip', 'sentence_id': 5, 'sz': 9},
             {'corpus_id': 'Ц.zip', 'sentence_id': 0, 'sz': 5}, {'corpus_id': 'Ц.zip', 'sentence_id': 1, 'sz': 7},
             {'corpus_id': 'Ц.zip', 'sentence_id': 2, 'sz': 9}, {'corpus_id': 'Ц.zip', 'sentence_id': 5, 'sz': 9}],
            rstat
        )



