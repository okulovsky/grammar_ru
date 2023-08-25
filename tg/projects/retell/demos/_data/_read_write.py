from pathlib import Path
from typing import *

sep = "<sep>"


def read_retell(path: Path) -> List[str]:
    with open(path, 'r') as file:
        return file.read().split(sep)


def write_retell(path: Path, text: str) -> None:
    with open(path, 'w') as file:
        for elem in text:
            file.write(f"{elem}{sep}")
