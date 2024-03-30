from typing import List, Optional, Any
from architecture_refactored import NlpAlgorithmRefactored, DataBundle, TokenData


class RepetitionsAlgorithmRefactored(NlpAlgorithmRefactored):
    def __init__(self,
                 vicinity: int = 50,
                 allow_simple_check=True,
                 allow_normal_form_check=True,
                 allow_tikhonov_check=True):
        self.vicinity = vicinity
        self.allow_simple_check = allow_simple_check
        self.allow_normal_form_check = allow_normal_form_check
        self.allow_tikhonov_check = allow_tikhonov_check
        self.pymorphy = PyMorphyFeaturizer() if self.allow_normal_form_check or self.allow_tikhonov_check else None
        self.tikhonov = MorphemeTikhonovFeaturizer(['ROOT']) if self.allow_tikhonov_check else None

    def generate_merge_index(self, tokens: List[TokenData]):
        merge_index = []
        for i, token_data_i in enumerate(tokens):
            for j, token_data_j in enumerate(tokens[max(0, i - self.vicinity):i]):
                if token_data_i.text.lower() == token_data_j.text.lower():
                    merge_index.append((i, j))
        return merge_index

    def _run_inner(self, data_bundle: DataBundle, index: List[int]) -> List[TokenData]:
        tokens = data_bundle.tokens
        merge_index = self.generate_merge_index(tokens)

        # Initialize all tokens with no error
        for token in tokens:
            token.error = False
            token.hint = ""
            token.error_type = ""

        # Check for repetitions using simple, normal form, and Tikhonov checks
        for i, j in merge_index:
            token_i = tokens[i]
            token_j = tokens[j]
            if self.allow_simple_check and token_i.text.lower() == token_j.text.lower():
                token_i.error = True
                token_i.hint = f"Simple repetition with '{token_j.text}' at position {j}"
                token_i.error_type = NlpAlgorithmRefactored.ErrorTypes['Stylistic']
            # Add checks for normal form and Tikhonov here, updating token_i as necessary

        # Filter and return only the tokens with errors
        return [token for token in tokens if token.error]

