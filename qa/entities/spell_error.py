from dataclasses import dataclass


@dataclass
class SpellError:
    original_word: str
    sentence: str
    suggestions: list[str]
