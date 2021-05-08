from typing import *
import pandas as pd
from .nlp_algorithm import NlpAlgorithm

class CombinedNlpAlgorithm(NlpAlgorithm):
    def __init__(self, algorithms: List[NlpAlgorithm]):
        super(CombinedNlpAlgorithm, self).__init__('status','suggestion')
        self._algorithms = algorithms

    def run(self, df):
        statuses = []
        suggestions = []
        for alg in self._algorithms:
            statuses.append(alg.get_status_column())
            suggestions.append(alg.get_suggest_column())
            alg.run(df)
        active = pd.Series(True, df.index)
        df['algorithm'] = None
        df['status'] = True
        df['suggestion'] = None
        for i in range(len(statuses)):
            df.loc[active,'status'] = df.loc[active, statuses[i]]
            current = active & ~df[statuses[i]]
            df.loc[current,'suggestion'] = df.loc[current, suggestions[i]]
            df.loc[current & ~df[statuses[i]], 'algorithm'] = self._algorithms[i].get_name()
            active = active & df[statuses[i]]





