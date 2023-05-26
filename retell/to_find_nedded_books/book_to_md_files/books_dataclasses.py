from dataclasses import dataclass


@dataclass
class ChapterInfo:
    name: str
    retell: str
    summary: str


@dataclass
class BookInfo:
    name: str
    chapters: list[ChapterInfo]
