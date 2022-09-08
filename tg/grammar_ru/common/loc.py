from pathlib import Path
import os
from ...common._common.locations import LocationsClass


class LocHolder(LocationsClass):
    def __init__(self):
        super(LocHolder, self).__init__()
        self.data_path = self.data_cache_path
        self.dependencies_path = self.data_path/'external_models'
        os.makedirs(self.dependencies_path, exist_ok=True)
        self.temp_path = self.data_path/'temp'
        os.makedirs(self.temp_path, exist_ok=True)

        self.corpus_path = self.data_path / 'corpus'
        self.processed_path = self.data_path / 'processed'
        self.raw_path = self.data_path / 'raw'
        self.bundles_path = self.data_path / 'bundles'
        self.task_states = self.data_path/'task_states'

        self.error_dumps = self.temp_path/'error_dumps'

        self.models_path = self.root_path/'models'

        self.grammar_ru_folder = self.root_path/'tg/grammar_ru'
        self.test_corpus_basic = self.grammar_ru_folder/'tests/samples/basic.zip'
        self.test_corpus_enriched = self.grammar_ru_folder/'tests/samples/enriched.zip'
        self.test_bundle = self.grammar_ru_folder/'tests/samples/bundle/'



Loc = LocHolder()