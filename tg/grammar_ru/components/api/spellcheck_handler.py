import uvicorn

from tg.grammar_ru.algorithms import spellcheck_refactored
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

    spell_checker = spellcheck_refactored.SpellcheckAlgorithmRefactored()

    response = spell_checker.run_on_string(text)
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=7040, log_level="debug")
