from fastapi import FastAPI

from ..services.response_constructor import ResponseConstructor
from ..services.model_messenger import ModelMessenger
from ..services.prompt_processor import PromptProcessor
from ..data_structures.api_data import RequestBody, ResponseBody

app = FastAPI()


@app.post(
    "/finalize-paragraph",
    response_model=ResponseBody,
    description="Finalize single paragraph chosen by user."
)
async def finalize_paragraph(finalize_request: RequestBody):
    prompt_processor = PromptProcessor()
    model_messenger = ModelMessenger()
    response_constructor = ResponseConstructor()

    prompt = prompt_processor.construct_prompt(finalize_request.paragraphs)
    finalized_paragraph = model_messenger.get_finalized_text(prompt)

    return response_constructor.construct_response(finalize_request.paragraphs, finalized_paragraph)
