import dataclasses

import typing


@dataclasses.dataclass
class ProcessedWord:
    index: int
    error: bool
    error_type: str
    suggest: typing.List[str]
    algorithm: str
    hint: typing.Optional[str] = None
