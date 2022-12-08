from abc import abstractmethod, ABC
from typing import List, Dict, Union
from pathlib import Path
import pandas as pd


class ITransfuseSelector(ABC):
    @abstractmethod
    def select(self, corpus: Path, df: pd.DataFrame, toc_row: Dict) -> Union[List[pd.DataFrame], pd.DataFrame]:
        raise NotImplementedError()
