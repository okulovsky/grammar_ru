from dataclasses import dataclass


@dataclass
class SpellError:
    word: str
    index: int
    start_position: int
    end_position: int
    suggestions: list[str]
