import pickle
import pandas as pd
from fastapi import FastAPI

from tg.grammar_ru.algorithms import NlpAlgorithm
from tg.grammar_ru.common import Loc, Separator, DataBundle
from tg.projects.punct.punct_algorithm import PunctNlpAlgorithm, PunctModelUnpickler


app = FastAPI()


def initialize_algorithms():
    with open('model.pkl', 'rb') as f_model, open('batcher.pkl', 'rb') as f_batcher:
        model = PunctModelUnpickler(f_model).load()
        batcher = pickle.load(f_batcher)

    punct_algorithm = PunctNlpAlgorithm(model, batcher, Loc.bundles_path/'punct/550k/sample_to_navec.parquet')

    return [punct_algorithm]


algorithms = initialize_algorithms()


def build_sentence_from_frame(df):
    return (
        df
        .assign(word_print=(
            df.word + pd.Series(' ', index=df.index) * df.word_tail
        ))
        .groupby('sentence_id').word_print
        .sum()
    )


def correct_syntax_errors(query_df, algo_result):
    syntax_errors = algo_result[algo_result[NlpAlgorithm.ErrorType] == NlpAlgorithm.ErrorTypes.Syntax]
    to_drop = syntax_errors[syntax_errors[NlpAlgorithm.Suggest] == 'no']
    to_correct = syntax_errors[syntax_errors[NlpAlgorithm.Suggest] != 'no']

    result = query_df.copy()
    result.loc[to_drop.index + 1, 'word'] = ''
    result.loc[to_correct.index + 1].word = to_correct[NlpAlgorithm.Suggest]

    return build_sentence_from_frame(result)


def correct_query(query: str) -> str:
    query_df = Separator.separate_string(query)
    db = DataBundle(src=query_df)
    combined_result = NlpAlgorithm.combine_algorithms(db, query_df.index, *algorithms)
    print(combined_result)

    return correct_syntax_errors(query_df, combined_result)


@app.get('/check')
def root(q: str = ''):
    corrected = correct_query(q)
    result = {
        'query': q,
        'corrected': corrected
    }

    return result
