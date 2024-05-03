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
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "paragraphs": [
                        {
                            "type": "plan",
                            "content": "Меня зовут Томас и мой основной",
                            "is_target": True,
                            "sub_paragraphs": [],
                        },
                        {
                            "type": "plan",
                            "content": "Меня зовут Томас и мой основной ",
                            "is_target": False,
                            "sub_paragraphs": [
                                {
                                    "type": "plan",
                                    "content": "Меня зовут Томас и мой основной ",
                                    "is_target": False,
                                    "sub_paragraphs": []
                                }
                            ],
                        }
                    ]
                }
            ]
        }
    }


class ResponseBody(BaseModel):
    paragraphs: List[Paragraph]
