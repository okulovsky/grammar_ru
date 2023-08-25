from typing import *


def _skip_space(s, ptr):
    end = len(s) - ptr
    for i in range(0, end):
        if not str.isspace(s[i + ptr]):
            return i
    return end


def _generate_offsets(string: str, substrings: List[str]):
    result = []
    total_length = 0

    ptr = _skip_space(string, 0)
    for s in substrings:
        if not string.startswith(s, ptr):
            raise ValueError(f'Missing substring {s} in {string} starting at {ptr}')
        ptr += len(s)
        offset = _skip_space(string, ptr)
        result.append(offset)
        ptr += offset
    return result, ptr
