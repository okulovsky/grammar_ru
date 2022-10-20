from dataclasses import dataclass
from datetime import datetime


@dataclass()
class Book:
    name: str
    rel_url: str
    publication_date: datetime = None
    content: str = None
    review_cnt = None
