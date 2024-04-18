import pytest
from ...services.model_messenger import ModelMessenger


@pytest.mark.parametrize(
    'passed_text, expected_text',
    [
        ('Awesome prompt', 'Awesome prompt'),
        ('One more awesome prompt', 'One more awesome prompt'),
    ]
)
def test_model_messenger(passed_text, expected_text):
    model_messenger = ModelMessenger()
    result = model_messenger.get_finalized_text(passed_text)
    assert result == expected_text

def test_empty_prompt():
    model_messenger = ModelMessenger()
    empty_prompt = ""
    
    with pytest.raises(ValueError) as verr:
        model_messenger.get_finalized_text(empty_prompt)

    assert verr.type is ValueError
    assert "Prompt is empty" == str(verr.value)