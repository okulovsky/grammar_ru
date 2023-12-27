import pandas as pd
import json
import os
import uuid

from typing import List, Tuple
from tg.grammar_ru import CorpusReader
from pathlib import Path
from tg.projects.book_fragments.eng_localizator import EnglishLocalizator

class FragmentsBuilder:
    def __init__(
        self,
        corpus: Path,
        words_limit=500,
        output_path="./fragments",
        file_name="fragments",
        localizator=EnglishLocalizator(),
        prompt="retell this text in simple sentences and simple words, divide every complex sentence into simple sentences, replace artistic words with simple ones, remove homogeneous parts: {}"
    ) -> None:
        self.corpus_reader = CorpusReader(corpus)
        self.words_limit = words_limit
        self.output_path = f'{output_path}/{file_name}.json'
        self.localizator = localizator
        self.prompt = prompt

        self.narrative_parts, self.dialog_parts = [], []
        self.narrative_parts_len, self.dialog_parts_len = 0, 0

        self.total_frames_num = len(list(self.corpus_reader.get_frames()))
        self.cur_frame_num = 0

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        self.dialog_sentence = False
        self.dialog_sentence_closed = False
        self.start_word_id = 0
        self.end_word_id = 0

        self.json_data = None

    def construct_fragments_json(self) -> None:
        self._create_file()
        for frame in self.corpus_reader.read_frames():
            self.cur_frame_num += 1
            print(
                "Processing frame with id: "
                f"{frame['file_id'].values[1]}, "
                f"{self.cur_frame_num}/{self.total_frames_num}",
                end="\r"
            )
            self._construct_paragraph(frame)

        self._write_file_json()

    def _construct_paragraph(self, frame: pd.DataFrame) -> None:
        prev_paragraph_id = frame['paragraph_id'].values[0]
        prev_paragraph = frame[frame['paragraph_id'] == prev_paragraph_id]
        for paragraph_id in frame['paragraph_id'].unique().tolist():
            cur_paragraph = frame[frame['paragraph_id'] == paragraph_id]
            paragraph_length = cur_paragraph.count().values[1]

            (
                (narrative_parts, narrative_parts_len), 
                (dialog_parts, dialog_parts_len)
            ) = self._construct_sentences(frame, paragraph_id)

            if (self.narrative_parts_len + narrative_parts_len > self.words_limit):
                self.end_word_id = prev_paragraph['word_id'].values[-1]

                self._write_fragment_to_data_json(
                    frame, "narrative", self.narrative_parts)
                self.start_word_id = self.end_word_id

                self.narrative_parts = []
                self.narrative_parts_len = 0

            self.narrative_parts_len += narrative_parts_len

            if len(narrative_parts) != 0:
                self.narrative_parts.append(narrative_parts) 
                self.narrative_parts.append('\n')

            if (self.dialog_parts_len + dialog_parts_len > self.words_limit):
                self.end_word_id = prev_paragraph['word_id'].values[-1]
                self._write_fragment_to_data_json(
                    frame, "dialog", self.dialog_parts)
                self.start_word_id = self.end_word_id

                self.dialog_parts = []
                self.dialog_parts_len = 0

            self.dialog_parts_len += dialog_parts_len

            if len(dialog_parts) != 0:
                self.dialog_parts.append(dialog_parts)
                self.dialog_parts.append('\n')

            prev_paragraph = cur_paragraph
        

    def _construct_sentences(
        self,
        frame: pd.DataFrame,
        paragraph_id: int
    ) -> Tuple[Tuple[List[str], int], Tuple[List[str], int]]:
        dialog_parts, narrative_parts = [], []
        dialog_parts_len, narrative_parts_len = 0, 0
        for sentence_id in frame[frame['paragraph_id'] == paragraph_id]['sentence_id'].unique().tolist():
            cur_sentence = self._construct_sentence(frame, sentence_id)
            if len(cur_sentence) != 0 and cur_sentence[-1][-1] == ' ':
                cur_sentence[-1] = cur_sentence[-1][:-1]

            if self.dialog_sentence:
                dialog_parts.append(''.join(cur_sentence))
                dialog_parts_len += len(cur_sentence)
                if self.dialog_sentence_closed:
                    self.dialog_sentence = False
                    self.dialog_sentence_closed = False
            else:
                narrative_parts.append(''.join(cur_sentence))
                narrative_parts_len += len(cur_sentence)

        return (' '.join(narrative_parts), narrative_parts_len), (' '.join(dialog_parts), dialog_parts_len)

    def _construct_sentence(self, frame: pd.DataFrame, sentence_id: int) -> List[str]:
        self.localizator.dialog_sentence = self.dialog_sentence
        self.localizator.dialog_sentence_closed = self.dialog_sentence_closed
        
        cur_sentence = self.localizator.construct_sentence(frame, sentence_id)
        self.dialog_sentence = self.localizator.dialog_sentence
        self.dialog_sentence_closed = self.localizator.dialog_sentence_closed
        
        return cur_sentence

    def _write_fragment_to_data_json(self, frame: pd.DataFrame, text_type: str, text: List[str]) -> None:
        self.json_data['fragments'].append(
            {
                "id": str(uuid.uuid1()),
                "file_id": frame["file_id"].values[0],
                "text_type": text_type,
                "text": ''.join(text),
                "retell": "",
                "word_start_id": int(self.start_word_id),
                "word_end_id": int(self.end_word_id)
            }
        )

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
