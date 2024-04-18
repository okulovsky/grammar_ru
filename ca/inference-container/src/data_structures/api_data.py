from enum import Enum
from typing import List
from pydantic import BaseModel

class ParagraphType(Enum):
    PLAN = "plan"
    FINAL = "final"

class Paragraph(BaseModel):
    type: ParagraphType
    content: str
    is_target: bool
    sub_paragraphs: List['Paragraph']

class RequestBody(BaseModel):
    paragraphs: List[Paragraph]

class ResponseBody(BaseModel):
    paragraphs: List[Paragraph]
