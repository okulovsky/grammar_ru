class ModelMessenger:
    def __init__(self) -> None:
        pass

    def get_finalized_text(self, prompt: str) -> str:
        if prompt == "":
            raise ValueError("Prompt is empty")

        return prompt