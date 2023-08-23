from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChapterInfo:
    name: str
    retell: str
    summary: Optional[str] = ''


@dataclass
class BookInfo:
    name: str
    chapters: List[ChapterInfo]
