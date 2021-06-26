from typing import *
from ...common import DataBundle
from .formats.interformat_parser import InterFormatParser
from .corpus_writer import CorpusWriter
from .corpus_reader import CorpusReader
from ..features import Enricher, Featurizer
from io import BytesIO
import shutil
from yo_fluq_ds import *
import zipfile

class _ParallelParser:
    def __init__(self, SRC, naming):
        self.SRC = SRC
        self.naming = naming

    def __call__(self, file):
        return InterFormatParser(self.SRC, file, self.naming).parse().to_list()

class _CorpusEnricher:
    def __init__(self, steps: List[Union[Enricher, Callable, Featurizer]]):
        self.steps = steps

    def __call__(self, db: DataBundle):
        for index, step in enumerate(self.steps):
            if isinstance(step, Enricher):
                step.enrich(db)
            elif isinstance(step, Featurizer):
                step.as_enricher().enrich(db)
            elif callable(step):
                result = step(db)
                if not result:
                    return None, None
            else:
                raise ValueError(f"The step {step} at index {index} is neither Enricher, Featurizer nor callable")
        return db.src.file_id.iloc[0], db

class CorpusBuilder:
    def __init__(self, corpus_folder, md_folder):
        self.corpus_folder = corpus_folder
        self.md_folder = md_folder


    def convert_interformat_folder_to_corpus(self, corpus_path, subfolder, naming, workers_count = None):
        corpus_path = self.corpus_folder / corpus_path
        subfolder = self.md_folder / subfolder
        writer = CorpusWriter(corpus_path, True)
        files = Query.folder(subfolder, '**/*.*').to_list()
        parser = _ParallelParser(self.md_folder, naming)
        query = Query.en(files)
        if workers_count is not None:
            query = query.parallel_select(parser, workers_count)
        else:
            query = query.select(parser)

        query.feed(fluq.with_progress_bar(total=len(files))).select_many(lambda z: z).foreach(writer.add_fragment)
        writer.finalize()


    def enrich_corpus(self, source, destination, steps: List[Union[Enricher,Callable, Featurizer]], workers = None):
        source = self.corpus_folder/source
        destination = self.corpus_folder/destination
        shutil.copy(source, destination)
        reader = CorpusReader(source)
        toc = reader.get_toc()
        good_ids = []
        writer = CorpusWriter(destination, True)

        query = reader.get_bundles(toc.index)
        enricher = _CorpusEnricher(steps)
        if workers is not None:
            query = query.parallel_select(enricher, workers)
        else:
            query = query.select(enricher)
        query = query.feed(fluq.with_progress_bar(total=toc.shape[0]))
        for result in query:
            uid, db = result
            if uid is None or db is None:
                continue
            good_ids.append(uid)
            writer.add_bundle(db)
        writer.finalize(toc.loc[toc.index.isin(good_ids)])



