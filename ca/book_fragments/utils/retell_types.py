import datetime
from typing import TypedDict


class Tags(TypedDict):
    id: str
    file_id: str
    text_type: str
    text: str
    retell: str
    word_start_id: int
    word_end_id: int

class RetellFragment(TypedDict):
    job_id: str
    tags: Tags
    # timestamp: datetime
    result: str