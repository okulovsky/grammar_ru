from pathlib import Path
import zipfile
from io import BytesIO
from yo_fluq_ds import *
from ...common import DataBundle, Separator

class ISrcReader:
    def read_frames(self):
        raise NotImplementedError()

class CorpusReader(ISrcReader):
    def __init__(self, location: Path):
        self.location = location


    def _read_frame(self, file, fname):
        buffer = BytesIO(file.read(fname))
        df = pd.read_parquet(buffer)
        return df


    def get_toc(self):
        with zipfile.ZipFile(self.location, 'r') as file:
            return self._read_frame(file,'toc.parquet')

    def read_toc(self):
        return self.get_toc()


    def _read_src(self, file, uid):
        df = self._read_frame(file, f'src/{uid}.parquet')
        #if self.id_shift > 0:
        #    for column in self.updatable_columns:
        #        df[column] += self.id_shift
        #    df.index = list(df.word_id)
        df['file_id'] = uid
        df['corpus_id'] = self.location.name
        Separator.validate(df)
        return df

    def _get_frames_iter(self, uids):
        with zipfile.ZipFile(self.location, 'r') as file:
            for uid in uids:
                yield  self._read_src(file, uid)


    def _get_custom_frames_iter(self, frame_type, uids):
        with zipfile.ZipFile(self.location, 'r') as file:
            for uid in uids:
                yield self._read_frame(file, f'{frame_type}/{uid}.parquet')



    def get_frames(self, uids = None, frame_type = None):
        if uids is None:
            uids = self.get_toc().index
        if frame_type is None:
            return Queryable(self._get_frames_iter(uids), len(uids))
        else:
            return Queryable(self._get_custom_frames_iter(frame_type, uids), len(uids))

    def read_frames(self, uids=None, frame_type=None):
        return self.get_frames(uids, frame_type)


    def _get_fearurizers_name(self, file):
        return Query.en(file.namelist()).where(lambda z: '/' in z and z.endswith('.parquet')).select(lambda z: z.split('/')[0]).distinct().where(lambda z: z!='src').to_list()


    def _get_bundles_iter(self, uids, toc):
        with zipfile.ZipFile(self.location, 'r') as file:
            featurizers = self._get_fearurizers_name(file)
            for uid in uids:
                frames = {}
                frames['src'] = self._read_src(file, uid)
                for featurizer in featurizers:
                    df = self._read_frame(file, f'{featurizer}/{uid}.parquet')
                    #if df.index.name not in self.updatable_columns:
                    #    raise ValueError(f'Featurizer {featurizer} has produced a DF with the index {df.index.name} which is not updatable\n{df.head()}')
                    #if self.id_shift > 0:
                    #    df.index = df.index+self.id_shift
                    frames[featurizer] = df
                bundle = DataBundle(**frames)

                bundle.additional_information.uid = uid
                for c in toc.columns:
                    bundle.additional_information[c] = toc.loc[uid, c]

                yield bundle


    def get_bundles(self, uids = None):
        toc = self.get_toc()
        if uids is None:
            uids = toc.index
        return Queryable(self._get_bundles_iter(uids, toc), len(uids))

    def read_bundles(self, uids=None):
        return self.get_bundles(uids)











