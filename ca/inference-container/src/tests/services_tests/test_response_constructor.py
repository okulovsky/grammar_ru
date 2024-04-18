from typing import List
import pytest

from ...services.response_constructor import ResponseConstructor
from ...data_structures.api_data import Paragraph, ParagraphType, ResponseBody


@pytest.mark.parametrize(
    'paragraphs, finalized_text, expexted_structure',
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
                    is_target=False, sub_paragraphs=[]
                )
            ],
            "Finalized text",
            ResponseBody(paragraphs=[
                Paragraph(
                    type=ParagraphType.FINAL,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False, sub_paragraphs=[]
                ),
                Paragraph(
                    type=ParagraphType.FINAL,
                    content="Finalized text",
                    is_target=False, sub_paragraphs=[]
                ),
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False, sub_paragraphs=[]
                )
            ]),
        )
    ]
)
def test_finalized_paragraph_plain_substitution(paragraphs: List[Paragraph], finalized_text: str, expexted_structure: ResponseBody):
    response_constructor = ResponseConstructor()
    result = response_constructor.construct_response(
        paragraphs, finalized_text)

    assert len(result.paragraphs) == len(expexted_structure.paragraphs)

    for i in range(len(result.paragraphs)):
        assert result.paragraphs[i] == expexted_structure.paragraphs[i]


@pytest.mark.parametrize(
    'paragraphs, finalized_text, expexted_structure',
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
                    is_target=False, sub_paragraphs=[
                        Paragraph(
                            type=ParagraphType.PLAN,
                            content="Inner paragraph",
                            is_target=True, sub_paragraphs=[]
                        )
                    ]
                ),
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False, sub_paragraphs=[]
                )
            ],
            "Finalized text",
            ResponseBody(paragraphs=[
                Paragraph(
                    type=ParagraphType.FINAL,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False, sub_paragraphs=[]
                ),
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                    is_target=False, sub_paragraphs=[
                        Paragraph(
                            type=ParagraphType.FINAL,
                            content="Finalized text",
                            is_target=False, sub_paragraphs=[]
                        )
                    ]
                ),
                Paragraph(
                    type=ParagraphType.PLAN,
                    content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    is_target=False, sub_paragraphs=[]
                )
            ]),
        )
    ]
)
def test_finalized_paragraph_tree_substitution(paragraphs: List[Paragraph], finalized_text: str, expexted_structure: ResponseBody):
    response_constructor = ResponseConstructor()
    result = response_constructor.construct_response(
        paragraphs, finalized_text)

    assert len(result.paragraphs) == len(expexted_structure.paragraphs)

    for i in range(len(result.paragraphs)):
        assert result.paragraphs[i] == expexted_structure.paragraphs[i]


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
                type=ParagraphType.PLAN,
                content="Donec et nisi vel massa sodales efficitur ac sed mi.",
                is_target=True, sub_paragraphs=[
                    Paragraph(
                        type=ParagraphType.PLAN,
                        content="Inner paragraph",
                        is_target=False, sub_paragraphs=[]
                    )
                ]
            ),
            Paragraph(
                type=ParagraphType.PLAN,
                content="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                is_target=False, sub_paragraphs=[]
            )
        ],
    )
)
def test_empty_finalized_paragraph(paragraphs: List[Paragraph]):
    response_constructor = ResponseConstructor()

    with pytest.raises(ValueError) as verr:
        response_constructor.construct_response(paragraphs, "")

    assert verr.type is ValueError
    assert "Finalized paragraph is empty" == str(verr.value)


def test_empty_paragraphs_list():
    response_constructor = ResponseConstructor()

    with pytest.raises(ValueError) as verr:
        response_constructor.construct_response([], "Finalized")

    assert verr.type is ValueError
    assert "Paragraphs list is empty" == str(verr.value)
