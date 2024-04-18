from fastapi.testclient import TestClient
from ...app import app
from ...data_structures.api_data import RequestBody, ResponseBody, Paragraph, ParagraphType

test_app = TestClient(app)


def test_finalize_paragraph_endpoint():
    request_body = RequestBody(
        paragraphs=[
            Paragraph(
                type=ParagraphType.PLAN,
                content="Text to finalize",
                is_target=True,
                sub_paragraphs=[]
            ),
            Paragraph(
                type=ParagraphType.FINAL,
                content="Other ready paragraph",
                is_target=False,
                sub_paragraphs=[]
            ),
        ]
    )

    finalize_response = test_app.post(
        "/finalize-paragraph",
        content=request_body.model_dump_json()
    )

    response_body = ResponseBody.model_validate_json(finalize_response.content)

    assert len(response_body.paragraphs) == 2
    result_paragraph = response_body.paragraphs[0]

    assert result_paragraph.type == ParagraphType.FINAL
    assert result_paragraph.content == "Text to finalize"
    assert len(result_paragraph.sub_paragraphs) == 0
