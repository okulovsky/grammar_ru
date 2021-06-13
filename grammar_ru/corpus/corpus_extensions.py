from typing import *
import shutil
from .corpus_reader import CorpusReader
import zipfile
from yo_fluq_ds import *
from io import BytesIO

class FeaturizerParallelSelector:
    def __init__(self, featurizers):
        self.featurizers = featurizers

    def __call__(self, df):
        return (df.file_id.iloc[0], {key: featurizer.create_features(df) for key, featurizer in self.featurizers.items()})


class CorpusExtensions:
    @staticmethod
    def compute_featurizers(source, destination, featurizers: Dict[str,Any], workers = 4):
        shutil.copy(source, destination)
        reader = CorpusReader(source)
        ids = reader.get_toc().index


        with zipfile.ZipFile(destination, 'a', zipfile.ZIP_DEFLATED) as file:
            query = (reader
             .get_frames(ids)
             .parallel_select(FeaturizerParallelSelector(featurizers), workers)
             .feed(fluq.with_progress_bar(total=len(ids))))
            for result in query:
                uid, dfs = result
                for key, df in dfs.items():
                    bytes = BytesIO()
                    df.to_parquet(bytes)
                    file.writestr(f'{key}/{uid}.parquet', bytes.getbuffer())

