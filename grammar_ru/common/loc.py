from pathlib import Path
import os


class LocHolder:
    def __init__(self):
        self.data_path = Path(__file__).parent.parent/'data'
        self.dependencies_path = self.data_path/'external_models'
        os.makedirs(self.dependencies_path, exist_ok=True)
        self.temp_path = self.data_path/'temp'
        os.makedirs(self.temp_path, exist_ok=True)


Loc = LocHolder()