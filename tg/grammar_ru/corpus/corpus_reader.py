from pathlib import Path
import zipfile
from io import BytesIO
import numpy as np
import pandas as pd
from yo_fluq_ds import *
from ..common import DataBundle, Separator
import deprecated

ParrallelUids = Union[List[Dict[str, Any]], List[List[str]]]
Uids = Iterable


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

    def _get_filtered_uids(self, uids):
        uids = np.array(uids)
        toc = self.get_toc()
        matched_uids = np.where(np.isin(uids, toc.index))[0]
        uids = uids[matched_uids]
        if len(uids) == 0:
            raise ValueError(
                "No uids were found. If you use Parallel Coprus - make sure that uids connected to subcoprus type.")
        return uids

    def get_toc(self):
        with zipfile.ZipFile(self.location, 'r') as file:
            return self._read_frame(file, 'toc.parquet')

    def get_relations(self) -> pd.DataFrame:
        relations: List[pd.DataFrame] = list()
        with zipfile.ZipFile(self.location) as file:
            for relation in [filename for filename in file.namelist() if filename.startswith('relation')]:
                relations.append(self._read_frame(file, relation))
        if len(relations) > 0:
            return pd.concat(relations)
        else:
            return pd.DataFrame([], columns=['file_1', 'file_2', 'relation_name'])

    def get_src(self, uids: ParrallelUids):
        uid = next(iter(uids))
        is_list, is_dict = isinstance(uid, list), isinstance(uid, dict)
        if is_list:
            uids = [self._get_filtered_uids(sub_uids) for sub_uids in uids]
        elif is_dict:
            uids = [dict(zip(sub_uids.keys(), self._get_filtered_uids(list(sub_uids.values())))) for sub_uids in uids]
        else:
            raise TypeError("Uids must be list[list[str]]] or list[dict[str,str]]")

        with zipfile.ZipFile(self.location, 'r') as file:
            if is_dict:
                for uid_dict in uids:
                    yield self._get_uids_iter(uid_dict.values(), file, uid_dict.keys())
            elif is_list:
                for uid_list in uids:
                    yield self._get_uids_iter(uid_list, file, None)

    def read_toc(self):
        return self.get_toc()

    def read_relations(self) -> pd.DataFrame:
        return self.get_relations()

    def read_src(self, uids: ParrallelUids):
        return self.get_src(uids)

    def _read_src(self, file, uid):
        df = self._read_frame(file, f'src/{uid}.parquet')
        # if self.id_shift > 0:
        #    for column in self.updatable_columns:
        #        df[column] += self.id_shift
        #    df.index = list(df.word_id)
        df['file_id'] = uid
        df['corpus_id'] = self.location.name
        Separator.validate(df)
        return df

    def _get_uids_iter(self, uids: List[str], file, sub_corpus_names: Optional[Iterable[str]] = None) -> Union[
        List[str], Dict[str, str]]:
        dfs = list()
        for uid in uids:
            df = self._read_src(file, uid)
            if df.shape[0] > 0:
                dfs.append(df)
        if sub_corpus_names is not None:
            dfs = dict(zip(sub_corpus_names, dfs))
        return dfs

    def _get_frames_iter(self, uids):
        with zipfile.ZipFile(self.location, 'r') as file:
            for uid in uids:
                df = self._read_src(file, uid)
                if df.shape[0] > 0:
                    yield df

    def _get_custom_frames_iter(self, frame_type, uids):
        with zipfile.ZipFile(self.location, 'r') as file:
            for uid in uids:
                yield self._read_frame(file, f'{frame_type}/{uid}.parquet')

    def get_frames(self, uids=None, frame_type=None):
        if uids is None:
            uids = self.get_toc().index
        uids = self._get_filtered_uids(uids)
        if frame_type is None:
            return Queryable(self._get_frames_iter(uids), len(uids))
        else:
            return Queryable(self._get_custom_frames_iter(frame_type, uids), len(uids))

    def read_frames(self, uids=None, frame_type=None):
        return self.get_frames(uids, frame_type)

    def read_mapping_data(self):
        archive = zipfile.ZipFile(self.location)
        for file in archive.namelist():
            if file.startswith('mapping'):
                return pd.read_parquet(archive.extract(file))

    def _get_fearurizers_name(self, file):
        return Query.en(file.namelist()).where(lambda z: '/' in z and z.endswith('.parquet')).select(
            lambda z: z.split('/')[0]).distinct().where(lambda z: z != 'src').to_list()

    def _get_bundles_iter(self, uids, toc):
        with zipfile.ZipFile(self.location, 'r') as file:
            featurizers = self._get_fearurizers_name(file)
            for uid in uids:
                frames = {}
                src = self._read_src(file, uid)
                if src.shape[0] == 0:
                    continue
                frames['src'] = src
                for featurizer in featurizers:
                    df = self._read_frame(file, f'{featurizer}/{uid}.parquet')
                    # if df.index.name not in self.updatable_columns:
                    #    raise ValueError(f'Featurizer {featurizer} has produced a DF with the index {df.index.name} which is not updatable\n{df.head()}')
                    # if self.id_shift > 0:
                    #    df.index = df.index+self.id_shift
                    frames[featurizer] = df
                bundle = DataBundle(**frames)

                bundle.additional_information.uid = uid
                for c in toc.columns:
                    bundle.additional_information[c] = toc.loc[uid, c]

                yield bundle

    def get_bundles(self, uids=None):
        toc = self.get_toc()
        if uids is None:
            uids = toc.index
        uids = self._get_filtered_uids(uids)
        return Queryable(self._get_bundles_iter(uids, toc), len(uids))

    def read_bundles(self, uids=None):
        return self.get_bundles(uids)

    @staticmethod
    def read_frames_from_several_corpora(sources: Union[Path, List[Path]]):
        if isinstance(sources, Path):
            sources = [sources]
        readers = [CorpusReader(s) for s in sources]
        total_length = sum([r.get_toc().shape[0] for r in readers])

        query = Query.en(readers).select_many(
            lambda x: x.get_frames()
        )
        return Queryable(query, total_length)


@deprecated.deprecated('Use CorpusReader.read_frames_from_several_corpora')
def read_data(sources: Union[Path, List[Path]]) -> List[pd.DataFrame]:
    if isinstance(sources, Path):
        sources = [sources]
    readers = [CorpusReader(s) for s in sources]
    total_length = sum([r.get_toc().shape[0] for r in readers])

    query = Query.en(readers).select_many(
        lambda x: x.get_frames()
    )
    return Queryable(query, total_length).feed(fluq.with_progress_bar(console=True))
