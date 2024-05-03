from fastapi import FastAPI
from contextlib import asynccontextmanager

from ..services.response_constructor import ResponseConstructor
from ..services.prompt_processor import PromptProcessor
from ..data_structures.api_data import RequestBody, ResponseBody
from ..services.model import Model


models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["rugpt3small_based_on_gpt2"] = Model()
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post(
    "/finalize-paragraph",
    response_model=ResponseBody,
    description="Finalize single paragraph chosen by user.",
)
async def finalize_paragraph(finalize_request: RequestBody):
    prompt_processor = PromptProcessor()
    response_constructor = ResponseConstructor()

    prompt = prompt_processor.construct_prompt(finalize_request.paragraphs)
    finalized_paragraph = models["rugpt3small_based_on_gpt2"].generate_text(prompt)

    return response_constructor.construct_response(finalize_request.paragraphs, finalized_paragraph)
