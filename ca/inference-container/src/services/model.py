from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os


class Model:
    def __init__(self, path='/models/rugpt3small_based_on_gpt2') -> None:
        print(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)

    def generate_text(self, prompt: str) -> str:
        if prompt == "":
            raise ValueError("Prompt is empty")

        inputs = self.tokenizer([prompt], return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            generation_config=GenerationConfig(
                max_new_tokens=20
            )
        )

        res = self.tokenizer.decode(outputs[0])

        return res


if __name__ == "__main__":
    os.chdir("./ca/inference-container/src/services")
    cwd = os.getcwd()

    model = Model()
    print(model.generate_text("Кеша - хорошее имя?"))
