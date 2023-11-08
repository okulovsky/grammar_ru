from typing import List
import enchant
from architecture_refactored import NlpAlgorithmRefactored, DataBundle, TokenData

d = enchant.Dict('ru_RU')
print(d.check('квота'))

class SpellcheckAlgorithmRefactored(NlpAlgorithmRefactored):
    def __init__(self):
        self.spellchecker = enchant.Dict('ru_RU')

    def _run_inner(self, data_bundle: DataBundle, index: List[int]) -> List[TokenData]:
        # Process each token based on the provided index
        for i in index:
            token_data = data_bundle.tokens[i]
            # Check if the token is a Russian word before checking spelling
            if token_data.type == 'ru':
                # Use the spellchecker to check the token
                if not self.spellchecker.check(token_data.text):
                    token_data.error = True
                    token_data.error_type = self.ErrorTypes['Orthographic']
                    token_data.suggestions = self.spellchecker.suggest(token_data.text)
        # Return the updated tokens
        return data_bundle.tokens

