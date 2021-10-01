from typing import *
import pandas as pd
from grammar_ru.algorithms import  NlpAlgorithm


class CombinedNlpAlgorithm(NlpAlgorithm):
    def __init__(self, algorithms: List[NlpAlgorithm]):
        super(CombinedNlpAlgorithm, self).__init__('status', 'suggestion')
        self._algorithms = algorithms

    def _run_inner(self, df):
        for alg in self._algorithms:
            alg.run(df)
        CombinedNlpAlgorithm.merge_algorithms(df, self._algorithms, 'status', 'suggestion')

    @staticmethod
    def merge_algorithms(df, algorithms, status_column='status', suggestion_column='suggestion', algorithm_column='algorithm'):
        statuses = []
        suggestions = []
        for alg in algorithms:
            statuses.append(alg.get_status_column())
            suggestions.append(alg.get_suggest_column())

        active = pd.Series(True, df.index)
        df[algorithm_column] = None
        df[status_column] = True
        df[suggestion_column] = None
        #pd.options.display.max_columns = None; pd.options.display.width = None; print(df)
        for i in range(len(statuses)):
            df.loc[active, status_column] = df.loc[active, statuses[i]]
            current = active & ~df[statuses[i]]
            if suggestions[i] is not None:
                df.loc[current, suggestion_column] = df.loc[current, suggestions[i]]
            df.loc[current & ~df[statuses[i]], algorithm_column] = algorithms[i].get_name()
            active = active & df[statuses[i]]
