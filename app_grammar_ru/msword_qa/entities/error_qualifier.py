from abc import ABC, abstractmethod


class ErrorQualifier(ABC):

    @staticmethod
    @abstractmethod
    def qualify(wrong_word: str, suggestions: list[str]) -> bool:
        raise NotImplementedError()


class TrueErrorQualifier(ErrorQualifier):

    @staticmethod
    def qualify(wrong_word: str, suggestions: list[str]) -> bool:
        return True


class TsaErrorQualifier(ErrorQualifier):

    @staticmethod
    def qualify(wrong_word: str, suggestions: list[str]) -> bool:
        if not (wrong_word.endswith("тся") or wrong_word.endswith("ться")):
            return False

        return (wrong_word.replace("тся", "ться") in suggestions or
                wrong_word.replace("ться", "тся") in suggestions)
