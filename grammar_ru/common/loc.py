from pathlib import Path
import os
from tg.common._common.locations import LocationsClass


class LocHolder(LocationsClass):
    def __init__(self):
        super(LocHolder, self).__init__()
        self.data_path = self.data_cache_path
        self.dependencies_path = self.data_path/'external_models'
        os.makedirs(self.dependencies_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)

        self.corpus_path = self.data_path / 'corpus'
        self.processed_path = self.data_path / 'processed'
        self.raw_path = self.data_path / 'raw'
        self.bundles_path = self.data_path / 'bundles'
        self.task_states = self.data_path/'task_states'

        self.error_dumps = self.temp_path/'error_dumps'

        self.models_path = self.root_path/'models'

        self.grammar_ru_folder = self.root_path/'grammar_ru'
        self.test_corpus_basic = self.root_path/'grammar_ru_tests/samples/basic.zip'
        self.test_corpus_enriched = self.root_path/'grammar_ru_tests/samples/enriched.zip'
        self.test_bundle = self.root_path/'grammar_ru_tests/samples/bundle/'



Loc = LocHolder()