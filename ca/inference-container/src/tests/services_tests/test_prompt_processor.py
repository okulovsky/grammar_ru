from typing import List
import pytest

from ...data_structures.api_data import Paragraph, ParagraphType
from ...services.prompt_processor import PromptProcessor


@pytest.mark.parametrize(
    'paragraphs, expected_prompt',
    [
        (
            [
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                    is_target=True, sub_paragraphs=[]
                )
            ],
            "Donec et nisi vel massa sodales efficitur ac sed mi."
        ),
        (
            [
                Paragraph(
                    type=ParagraphType.FINAL,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False, sub_paragraphs=[]
                ),
                Paragraph(
                    type=ParagraphType.FINAL,
                    content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                    is_target=False, sub_paragraphs=[]
                ),
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                    is_target=True, sub_paragraphs=[]
                )
            ],
            "Donec et nisi vel massa sodales efficitur ac sed mi."
        )
    ]
)
def test_correct_simple_prompt_construction(paragraphs: List[Paragraph], expected_prompt: str):
    prompt_processor = PromptProcessor()
    result = prompt_processor.construct_prompt(paragraphs)

    assert result == expected_prompt


@pytest.mark.parametrize(
    'paragraphs, expected_prompt',
    [
        (
            [
                Paragraph(
                    type=ParagraphType.FINAL,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False, sub_paragraphs=[]
                ),
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                    is_target=True, sub_paragraphs=[]
                ),
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=True, sub_paragraphs=[]
                )
            ],
            "Donec et nisi vel massa sodales efficitur ac sed mi.\nLorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
    ]
)
def test_multiple_paragraphs_prompt_contruction(paragraphs: List[Paragraph], expected_prompt: str):
    prompt_processor = PromptProcessor()
    result = prompt_processor.construct_prompt(paragraphs)

    assert result == expected_prompt


@pytest.mark.parametrize(
    'paragraphs, expected_prompt',
    [
        (
            [
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False,
                    sub_paragraphs=[
                        Paragraph(
                            type=ParagraphType.PLAN,
                            content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                            is_target=True, sub_paragraphs=[]
                        ),
                        Paragraph(
                            type=ParagraphType.PLAN,
                            content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                            is_target=False, sub_paragraphs=[]
                        )
                    ]
                )
            ],
            "Donec et nisi vel massa sodales efficitur ac sed mi."
        )
    ]
)
def test_target_nested_paragraph_prompt_contruction(paragraphs: List[Paragraph], expected_prompt: str):
    prompt_processor = PromptProcessor()
    result = prompt_processor.construct_prompt(paragraphs)

    assert result == expected_prompt


@pytest.mark.parametrize(
    'paragraphs, expected_prompt',
    [
        (
            [
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=True,
                    sub_paragraphs=[
                        Paragraph(
                            type=ParagraphType.PLAN,
                            content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                            is_target=True, sub_paragraphs=[]
                        ),
                        Paragraph(
                            type=ParagraphType.PLAN,
                            content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                            is_target=True, sub_paragraphs=[]
                        )
                    ]
                )
            ],
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\nDonec et nisi vel massa sodales efficitur ac sed mi.\nLorem ipsum dolor sit amet, consectetur adipiscing elit."
        )
    ]
)
def test_nested_paragraphs_prompt_contruction(paragraphs: List[Paragraph], expected_prompt: str):
    prompt_processor = PromptProcessor()
    result = prompt_processor.construct_prompt(paragraphs)

    assert result == expected_prompt


@pytest.mark.parametrize(
    'paragraphs',
    (
        [
            Paragraph(
                type=ParagraphType.FINAL,
                content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                is_target=False, sub_paragraphs=[]
            ),
            Paragraph(
                type=ParagraphType.FINAL,
                content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                is_target=True, sub_paragraphs=[]
            ),
            Paragraph(
                type=ParagraphType.PLAN,
                content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                is_target=True, sub_paragraphs=[]
            )
        ],
    )
)
def test_target_paragraph_with_incorrect_type(paragraphs: List[Paragraph]):
    prompt_processor = PromptProcessor()

    with pytest.raises(ValueError) as verr:
        prompt_processor.construct_prompt(paragraphs)

    assert verr.type is ValueError
    assert "Target paragraph type must be 'plan'" == str(verr.value)


def test_empty_paragraphs_list():
    prompt_processor = PromptProcessor()
    empty_paragraphs_list = []

    with pytest.raises(ValueError) as verr:
        prompt_processor.construct_prompt(empty_paragraphs_list)

    assert verr.type is ValueError
    assert "Paragraphs list is empty" == str(verr.value)


@pytest.mark.parametrize(
    'paragraphs',
    (
        [
            Paragraph(
                type=ParagraphType.FINAL,
                content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                is_target=False, sub_paragraphs=[]
            ),
            Paragraph(
                type=ParagraphType.FINAL,
                content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                is_target=False, sub_paragraphs=[]
            ),
            Paragraph(
                type=ParagraphType.PLAN,
                content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                is_target=False, sub_paragraphs=[]
            )
        ],
    )
)
def test_paragraphs_with_no_target(paragraphs: List[Paragraph]):
    prompt_processor = PromptProcessor()

    with pytest.raises(Exception) as verr:
        prompt_processor.construct_prompt(paragraphs)

    assert verr.type is Exception
    assert "There are no target paragraphs" == str(verr.value)


@pytest.mark.parametrize(
    'paragraphs',
    (
        [
            Paragraph(
                type=ParagraphType.FINAL,
                content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                is_target=False, sub_paragraphs=[
                    Paragraph(
                        type=ParagraphType.FINAL,
                        content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                        is_target=False, sub_paragraphs=[]
                    ),
                ]
            ),
            Paragraph(
                type=ParagraphType.PLAN,
                content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                is_target=True, sub_paragraphs=[]
            )
        ],
    )
)
def test_final_paragraph_contain_no_subparagraphs(paragraphs: List[Paragraph]):
    prompt_processor = PromptProcessor()

    with pytest.raises(Exception) as verr:
        prompt_processor.construct_prompt(paragraphs)

    assert verr.type is Exception
    assert "Final paragraph contains no subparagraphs" == str(verr.value)
