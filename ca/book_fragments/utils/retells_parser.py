from typing import Iterable, List
import pandas as pd
import itertools
from pathlib import Path
from retell_types import RetellFragment
from grammar_ru.common import Separator
from grammar_ru.corpus.corpus_writer import CorpusWriter


class RetellsParser:
    def __init__(self) -> None:
        pass

    def __prettify_paragraph(self, paragraph: str):
        paragraph_points = paragraph.split('\n')

        result_points = []
        for paragraph_point in paragraph_points:
            space_idx = paragraph_point.find(" ")
            if 0 < space_idx and space_idx < len(paragraph_point):
                if paragraph_point[space_idx - 1] == '.':
                    paragraph_point = paragraph_point[space_idx + 1:]

            if len(paragraph_point) == 0:
                continue

            try:
                if paragraph_point[-1] != '.':
                    paragraph_point += '.'
                result_points.append(paragraph_point)
            except Exception as ex:
                print(
                    f"Error occured in paragraph point: {paragraph_point} {ex}")

        return ' '.join(result_points)

    def __get_clean_text(self, raw_paragraphs: Iterable[RetellFragment]) -> List[str]:
        result = []
        for raw_paragraph in sorted(raw_paragraphs, key=lambda x: x['tags']['word_start_id']):
            result.append(self.__prettify_paragraph(raw_paragraph['result']))
        return result

    def parse_pickle_to_corpus(self, pickle_path: Path, corpus_path: Path):
        pickle_data = pd.read_pickle(pickle_path)
        records = pickle_data["records"]

        corpus_writer = CorpusWriter(corpus_path)

        for (file_name, file_group) in itertools.groupby(records, key=lambda x: x['tags']['file_id']):
            clean_text = self.__get_clean_text(file_group)
            file_df = Separator.separate_paragraphs(clean_text)
            corpus_writer.add_fragment(file_df, file_name=file_name)

        corpus_writer.finalize()
