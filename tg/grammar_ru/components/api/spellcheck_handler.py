import uvicorn
import pandas as pd

from .....tg.grammar_ru.algorithms import repetitions
from .....tg.grammar_ru.algorithms import spellcheck
from .....tg.grammar_ru.components.entities.processed_word import ProcessedWord
from fastapi import FastAPI, status
from pydantic import BaseModel

app = FastAPI(
    version='2.0',
)


class Request(BaseModel):
    text: str


@app.post("/check_on_orthographic",
          description="Check text on orthographic",
          status_code=status.HTTP_200_OK,
          )
def _handle(request: Request):
    text: str = request.text

    #alternative_algorithm = alternative.AlternativeAlgorithm()
    spell_checker = spellcheck.SpellcheckAlgorithm()
    repetitions_checker = repetitions.RepetitionsAlgorithm()

    algorithms = [spell_checker, repetitions_checker]

    result = spell_checker.new_run_on_string_multiple_algorithms(text, algorithms)
    # result: pd.DataFrame = spell_checker.run_on_string(text)
    # transpose_res = result.transpose().to_dict()
    #
    # response = [
    #     ProcessedWord(
    #         index=int(word),
    #         error=transpose_res[word]['error'],
    #         error_type=transpose_res[word]['error_type'],
    #         suggest=transpose_res[word]['suggest'],
    #         algorithm=transpose_res[word]['algorithm'],
    #         hint=transpose_res[word]['hint'],
    #     )
    #     for word in transpose_res
    # ]

    return result


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=7040, log_level="debug")
