from typing import List
from ..data_structures.api_data import Paragraph, ParagraphType
from .adapters.simple_adapter import SimpleAdapter


class PromptProcessor:
    def __init__(self, adapter=SimpleAdapter()) -> None:
        self.__adapter = adapter

    def construct_prompt(self, paragraphs: List[Paragraph]) -> str:
        if len(paragraphs) == 0:
            raise ValueError("Paragraphs list is empty")

        unsqueezed_paragraphs = []
        self.__unsqueeze_paragraphs(unsqueezed_paragraphs, paragraphs)

        target_paragraphs = self.__get_target_paragraphs(unsqueezed_paragraphs)
        target_text = self.__join_paragraphs(target_paragraphs)
        prompt = self.__adapter.construct_prompt(target_text)

        return prompt

    def __unsqueeze_paragraphs(self, result: List[Paragraph], paragraphs: List[Paragraph]) -> List[Paragraph]:
        for p in paragraphs:
            result.append(p)

            if p.type == ParagraphType.FINAL and len(p.sub_paragraphs) != 0:
                raise Exception("Final paragraph contains no subparagraphs")

            self.__unsqueeze_paragraphs(result, p.sub_paragraphs)

    def __get_target_paragraphs(self, paragraphs: List[Paragraph]) -> List[Paragraph]:
        target_paragraphs = []

        for p in paragraphs:
            if p.is_target:
                if p.type != ParagraphType.PLAN:
                    raise ValueError("Target paragraph type must be 'plan'")
                target_paragraphs.append(p)

        if len(target_paragraphs) == 0:
            raise Exception("There are no target paragraphs")

        return [p for p in paragraphs if p.is_target]

    def __join_paragraphs(self, paragraphs: List[Paragraph]) -> str:
        return '\n'.join([paragraph.content for paragraph in paragraphs])
