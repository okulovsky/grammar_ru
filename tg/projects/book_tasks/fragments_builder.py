import pandas as pd
import numpy as np
import json
import os
import uuid

from typing import List, Tuple
from tg.grammar_ru import CorpusReader
from pathlib import Path

from tg.grammar_ru.corpus.corpus_writer import CorpusFragment, CorpusWriter


class FragmentsBuilder:
    def __init__(
        self,
        corpus_reader: CorpusReader,
        words_limit=500,
        output_path="./fragments",
        file_name="fragments",
        prompt="retell this text in simple sentences and simple words, divide every complex sentence into simple sentences, replace artistic words with simple ones, remove homogeneous parts: {}"
    ) -> None:
        self.corpus_reader = corpus_reader
        self.words_limit = words_limit
        self.output_path = f'{output_path}/{file_name}'
        self.file_name = file_name
        self.prompt = prompt

        self.narrative_parts = []
        self.narrative_parts_len = 0
        self.dialog_parts = []
        self.dialog_parts_len = 0

        self.total_frames_num = len(list(self.corpus_reader.get_frames()))
        self.cur_frame_num = 0

        if not os.path.isdir("fragments"):
            os.mkdir("fragments")

        self.dialog_sentence = False
        self.start_word_id = 0
        self.end_word_id = 0

        self.json_data = None
        self._write_fragment_to_data = None
        self.corpus_writer = None

    def construct_fragments_json(self) -> None:
        self.output_path += '.json'
        self._write_fragment_to_data = self._write_fragment_to_data_json

        self._create_file()
        for frame in self.corpus_reader.read_frames().take(1):
            self.cur_frame_num += 1
            print(
                "Processing frame with id: "
                f"{frame['file_id'].values[1]}, "
                f"{self.cur_frame_num}/{self.total_frames_num}",
                end="\r"
            )
            self._construct_paragraph(frame)

        self._write_file_json()

    def construct_fragments_corpus(self) -> None:
        self._write_fragment_to_data = self._write_fragment_data_to_corpus
        self.corpus_writer = CorpusWriter(Path(f"./files/{self.file_name}.base.zip"))

        for frame in self.corpus_reader.read_frames().take(1):
            self.cur_frame_num += 1
            print(
                "Processing frame with id: "
                f"{frame['file_id'].values[1]}, "
                f"{self.cur_frame_num}/{self.total_frames_num}",
                end="\r"
            )
            self._construct_paragraph(frame)
        self.corpus_writer.finalize()


    def _construct_paragraph(self, frame: pd.DataFrame) -> None:
        prev_paragraph = frame[frame['paragraph_id'] == 0]
        for paragraph_id in frame['paragraph_id'].unique().tolist():
            cur_paragraph = frame[frame['paragraph_id'] == paragraph_id]
            paragraph_length = cur_paragraph.count().values[1]

            if (self.narrative_parts_len + paragraph_length > self.words_limit or
                    self.dialog_parts_len + paragraph_length > self.words_limit):
                self.end_word_id = prev_paragraph['word_id'].values[-1]
                self._write_fragment_to_data(frame)

                self.start_word_id = self.end_word_id
                self.narrative_parts_len = 0
                self.dialog_parts_len = 0

                self.narrative_parts = []
                self.dialog_parts = []

            self._construct_sentences(frame, paragraph_id)
            prev_paragraph = cur_paragraph
    
    def _construct_sentences(
        self,
        frame: pd.DataFrame,
        paragraph_id: int
    ) -> None:
        for sentence_id in frame[frame['paragraph_id'] == paragraph_id]['sentence_id'].unique().tolist():
            cur_sentence = self._construct_sentence(frame, sentence_id)

            if self.dialog_sentence:
                self.dialog_parts_len += len(cur_sentence)
                self.dialog_parts.append(''.join(cur_sentence))
                self.dialog_sentence = False
            else:
                self.narrative_parts_len += len(cur_sentence)
                self.narrative_parts.append(''.join(cur_sentence))

    def _construct_sentence(self, frame: pd.DataFrame, sentence_id: int) -> List[str]:
        self.dialog_sentence = False
        cur_sentence = []

        for word_id in frame[frame['sentence_id'] == sentence_id]['word_id'].unique().tolist():
            word = frame[frame['word_id'] == word_id]['word'].values[0]

            if word == '\u00A0':
                word = '\u0020'

            cur_sentence.append(
                word + ' ' * frame[frame['word_id']
                                   == word_id]['word_tail'].values[0]
            )

            if frame[frame['word_id'] == word_id]['word'].values[0] == '\u201c':
                self.dialog_sentence = True
        return cur_sentence

    def _write_fragment_to_data_json(self, frame: pd.DataFrame) -> None:
        self.json_data['fragments'].append(
            {
                "id": str(uuid.uuid1()),
                "file_id": frame["file_id"].values[0],
                "text_narrative": ''.join(self.narrative_parts),
                "text_dialog": ''.join(self.dialog_parts),
                "retell": "",
                "word_start_id": int(self.start_word_id),
                "word_end_id": int(self.end_word_id)
            }
        )
    
    def _write_fragment_data_to_corpus(self, frame: pd.DataFrame):
        df = pd.DataFrame({
                "id": [str(uuid.uuid1())],
                "file_id": [frame["file_id"].values[0]],
                "text_narrative": [''.join(self.narrative_parts)],
                "text_dialog": [''.join(self.dialog_parts)],
                "retell": [""],
                "word_start_id": [int(self.start_word_id)],
                "word_end_id": [int(self.end_word_id)],
            })

        df["word_id"] = range(1, len(df) + 1)
        df["sentence_id"] = range(1, len(df) + 1)
        df["paragraph_id"] = range(1, len(df) + 1)
        
        df.insert(2, "prompt", self.prompt)
        df.insert(2, "word_length", 0)
        df.insert(2, "word", 0)

        self.corpus_writer.add_fragment(CorpusFragment("crime_and_punishment", self.cur_frame_num, df, {}))


    def _create_file(self):
        with open(Path(self.output_path), 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "prompt": self.prompt,
                "fragments": []
            }))

        with open(Path(self.output_path), 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)


    def _write_file_json(self) -> None:
        with open(Path(self.output_path), 'w', encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=2)
