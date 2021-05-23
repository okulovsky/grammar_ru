from typing import *
import pandas as pd
from .architecture import validations
from .architecture import Separator
from nerus import NerusDoc, load_nerus
from functools import reduce


def get_text_from_nerus_doc(doc: NerusDoc) -> str:
    return reduce(lambda x, y: x + " " + y, map(lambda x: x.text, doc.sents))


def make_dataframe_from_nerus(doc_amount: int, skip: int = 0) -> pd.DataFrame:
    docs = load_nerus('../analyzers/natasha/models/nerus_lenta.conllu.gz')

    paragraphs = []

    for i in range(doc_amount + skip):
        doc = next(docs, None)

        if i < skip:
            continue

        # No docs are left
        if not doc:
            break

        paragraphs.append(get_text_from_nerus_doc(doc))

    return Separator.separate_paragraphs(paragraphs)


def create_chunks_from_dataframe(df: pd.DataFrame) -> List[List[str]]:
    validations.ensure_df_contains(validations.WordCoordinates, df)
    chunks = []

    for index, row in df.iterrows():
        if row['sentence_id'] == len(chunks):
            chunks.append([])

        chunks[row['sentence_id']].append(row['word'])

    return chunks
