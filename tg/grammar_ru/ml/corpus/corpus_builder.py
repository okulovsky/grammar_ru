from typing import *
from ...common import DataBundle, Loc
from .formats.interformat_parser import InterFormatParser
from .corpus_writer import CorpusWriter
from .corpus_buffered_writer import CorpusBufferedWriter
from .corpus_reader import CorpusReader
from ..features import Featurizer
from io import BytesIO
import shutil
from yo_fluq_ds import *
import zipfile
import traceback
from pathlib import Path
import datetime
from ....common._common import Logger

class _ParallelParser:
    def __init__(self, SRC, naming):
        self.SRC = SRC
        self.naming = naming

    def __call__(self, file):
        return InterFormatParser(self.SRC, file, self.naming).parse().to_list()

class _Corpusfeaturizer:
    def __init__(self,
                 steps: List[Union[Callable, Featurizer]],
                 required_uids: List[str]
                 ):
        self.steps = steps
        self.reqiored_uids = required_uids

    def __call__(self, db: DataBundle):
        uid = db.src.file_id.iloc[0]
        Logger.info(f'Processing {uid} at #{self.reqiored_uids.index(uid)}, total {len(self.reqiored_uids)} ')
        try:
            for index, step in enumerate(self.steps):
                Logger.info(type(step))
                if isinstance(step, Featurizer):
                    step.featurize(db)
                elif callable(step):
                    result = step(db)
                    if not result:
                        return None, None
                else:
                    raise ValueError(f"The step {step} at index {index} is neither featurizer, Featurizer nor callable")
            return uid, db
        except:
            Logger.error(f'Error in {uid}, {traceback.format_exc()}')
            return None, None


class CorpusBuilder:
    @staticmethod
    def convert_interformat_folder_to_corpus(
            corpus_path: Path,
            md_folder: Path,
            naming,
            workers_count = None):
        subfolder = md_folder
        writer = CorpusWriter(corpus_path, True)
        files = Query.folder(subfolder, '**/*.*').to_list()
        parser = _ParallelParser(md_folder, naming)
        query = Query.en(files)
        if workers_count is not None:
            query = query.parallel_select(parser, workers_count)
        else:
            query = query.select(parser)

        query.feed(fluq.with_progress_bar(total=len(files))).select_many(lambda z: z).foreach(writer.add_fragment)
        writer.finalize()

    @staticmethod
    def featurize_corpus(
                      source: Path,
                      destination: Path,
                      steps: List[Union[Callable, Featurizer]],
                      workers = None,
                      append = True,
                      uid_black_list = None
                      ):
        final_destination = destination
        reader = CorpusReader(source)
        toc = reader.get_toc()
        temp_destination = destination.parent/(destination.name+'.files')
        dst_toc_file = temp_destination / 'toc.parquet'
        seen_files = []
        if not append:
            shutil.rmtree(temp_destination, True)
        else:
            if os.path.isfile(dst_toc_file):
                old_toc = pd.read_parquet(dst_toc_file)
                seen_files = list(old_toc.index)
        os.makedirs(temp_destination, exist_ok=True)

        good_ids = []
        required_uids = [c for c in toc.index if c not in seen_files]
        if uid_black_list is not None:
            required_uids = [c for c in required_uids if c not in uid_black_list]
        query = reader.get_bundles(required_uids)
        featurizer = _Corpusfeaturizer(steps, list(required_uids))
        if workers is not None:
            query = query.parallel_select(featurizer, workers)
        else:
            query = query.select(featurizer)
        for result in query:
            uid, db = result
            if uid is None or db is None:
                continue
            good_ids.append(uid)
            for frame_name, df in db.data_frames.items():
                fname = temp_destination/frame_name/(uid+'.parquet')
                os.makedirs(fname.parent, exist_ok=True)
                df.to_parquet(fname)

            write_toc = toc.loc[toc.index.isin(good_ids+seen_files)]
            write_toc.to_parquet(dst_toc_file)

        CorpusWriter.collect_from_files(temp_destination, final_destination)
        shutil.rmtree(temp_destination)


    @staticmethod
    def assemble(
                 corpus_path: Path,
                 bundle_path: Path,
                 limit_entries: Optional[int] = None,
                 random_state: Optional[int] = None
                 ):
        reader = CorpusReader(corpus_path)
        toc = reader.get_toc()
        if limit_entries is not None:
            if random_state is None:
                uids = toc.iloc[:limit_entries].index
            else:
                uids = toc.sample(limit_entries, random_state=random_state).index
        else:
            uids = toc.index
        uids = list(uids)
        frames = list(reader.get_bundles().first().data_frames)
        os.makedirs(bundle_path, exist_ok=True)
        for frame in frames:
            df = pd.concat(reader.get_frames(uids, frame).feed(fluq.with_progress_bar()).to_list())
            df.to_parquet(bundle_path / f'{frame}.parquet')



    @staticmethod
    def transfuse_corpus(
            sources: Union[Path,List[Path]],
            destination: Path,
            words_per_frame: int = 50000,
            words_limit: Optional[int] = None,
            selector: Optional[Callable[[pd.DataFrame], Union[pd.DataFrame, List[pd.DataFrame]]]] = None
    ):

        if isinstance(sources, Path):
            sources = [sources]
        elif isinstance(sources, list):
            pass
        else:
            raise ValueError(f'`sources` have to be Path or list of Path, but was {type(sources)}')

        writer = CorpusBufferedWriter(
            destination,
            words_per_frame=words_per_frame,
            break_down_by_sentence=True
        )
        writer.open()
        readers = [CorpusReader(s) for s in sources]
        total_length = sum([r.get_toc().shape[0] for r in readers])

        query = Queryable(
            Query.en(readers).select_many(lambda x: x.get_frames()), 
            total_length
        )

        word_count = 0
        for index, frame in enumerate(query):
            if selector is None:
                dfs = [frame]
            else:
                dfs = selector(frame)
                if isinstance(dfs, pd.DataFrame):
                    dfs = [dfs]
                elif isinstance(dfs, list):
                    pass
                else:
                    raise ValueError(f'The output of `selector` is expected to be pd.DataFrame or List of them, but was {type(dfs)}')
            for df in dfs:
                try:
                    writer.add(df)
                except:
                    dump = dict(
                        input_frame = frame,
                        dfs = dfs,
                        failed_result = df
                    )
                    dump_file = Loc.error_dumps/f'{datetime.datetime.now}_corpus_builder_transfuse_corpus'
                    os.makedirs(dump_file.parent, exist_ok=True)
                    FileIO.write_pickle(dump, dump_file)
                    raise
                word_count += df.shape[0]
                Logger.info(f'Processed {word_count} words. {index}/{total_length}')
            if words_limit is not None and word_count>words_limit:
                break
        writer.close()

