from typing import List, Optional
from models import TokenData, DataBundle

class NlpAlgorithmRefactored:
    """Represents a base class for NLP algorithms that process text using dataclasses instead of DataFrames."""
    ErrorTypes = {
        'Unknown': 'unknown',
        'Orthographic': 'orthographic',
        'Grammatic': 'grammatic',
        'Stylistic': 'stylistic'
    }

    def _run_inner(self, data_bundle: DataBundle, index: List[int]) -> Optional[List[TokenData]]:
        """This method should be implemented by subclasses to process the tokens."""
        raise NotImplementedError()

    def _post_check(self, tokens: List[TokenData], algorithm_name: str) -> List[TokenData]:
        """Performs post-processing checks on the tokens."""
        for token_data in tokens:
            if token_data.error and not token_data.hint:
                token_data.hint = f"Error detected by {algorithm_name}"
            if not token_data.error_type:
                token_data.error_type = self.ErrorTypes['Unknown']
        return tokens

    def run(self, data_bundle: DataBundle) -> List[TokenData]:
        """Executes the NLP algorithm on the provided data."""
        result_tokens = self._run_inner(data_bundle, [i for i, _ in enumerate(data_bundle.tokens)])
        if result_tokens is None:
            result_tokens = [TokenData(text=t.text, type=t.type) for t in data_bundle.tokens]
        return self._post_check(result_tokens, type(self).__name__)

    def run_on_string(self, s: str) -> List[TokenData]:
        """Executes the NLP algorithm on a given string."""
        # Tokenize the string and create a DataBundle
        tokens = [TokenData(text=word, type="word") for word in s.split()]
        data_bundle = DataBundle(tokens=tokens)
        return self.run(data_bundle)

    @staticmethod
    def combine(token_lists: List[List[TokenData]]) -> List[TokenData]:
        """Combines the results from multiple token lists into a single list."""
        combined_tokens = []
        for tokens in token_lists:
            for token in tokens:
                existing_token = next((t for t in combined_tokens if t.text == token.text), None)
                if existing_token:
                    existing_token.error |= token.error
                    existing_token.suggestions.extend(token.suggestions)
                    existing_token.hint = token.hint or existing_token.hint
                    existing_token.error_type = token.error_type or existing_token.error_type
                else:
                    combined_tokens.append(token)
        return combined_tokens

    @staticmethod
    def combine_algorithms(data_bundle: DataBundle, algorithms: List['NlpAlgorithmRefactored']) -> List[TokenData]:
        """Runs multiple NLP algorithms on the data and combines their results."""
        results = [algorithm.run(data_bundle) for algorithm in algorithms]
        return NlpAlgorithmRefactored.combine(results)
