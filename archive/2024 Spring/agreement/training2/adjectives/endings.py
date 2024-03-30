import numpy as np

NEW = {"ая", "ого", "ое", "ой", "ом", "ому", "ую", "ые", "ый", "ым", "ыми", "ых"}

GOOD = {
    "ая",
    "его",
    "ее",
    "ей",
    "ем",
    "ему",
    "ие",
    "ий",
    "им",
    "ими",
    "их",
    "ую",
    "яя",
    "юю",
    "ого",
    "ое",
    "ой",
    "ому",
    "ом",
}  # легкий

BIG = {
    "ая",
    "ие",
    "им",
    "ими",
    "их",
    "ого",
    "ое",
    "ой",
    "ом",
    "ому",
    "ую",
    "ые",
    "ым",
    "ыми",
    "ых",
}  # золотой

NEW_list = sorted(list(NEW))
GOOD_list = sorted(list(GOOD))
BIG_list = sorted(list(BIG))

ALL_ENDS_list = NEW_list + GOOD_list + BIG_list
POSSIBLE_ENDINGS = set(ALL_ENDS_list)
endings_nums = {e: i for i, e in enumerate(ALL_ENDS_list)}

NEW_num_by_end = {e: i for i, e in enumerate(NEW_list)}
GOOD_num_by_end = {e: i + len(NEW_num_by_end) for i, e in enumerate(GOOD_list)}
BIG_num_by_end = {
    e: i + len(NEW_num_by_end) + len(GOOD_num_by_end) for i, e in enumerate(BIG_list)
}

nums_by_decl_and_end = (
    {("new", e): n for e, n in NEW_num_by_end.items()}
    | {("good", e): n for e, n in GOOD_num_by_end.items()}
    | {("big", e): n for e, n in BIG_num_by_end.items()}
)


def extract_ending(word: str):
    for possible_ending in POSSIBLE_ENDINGS:
        if word.lower().endswith(possible_ending):
            return possible_ending
    return np.nan
