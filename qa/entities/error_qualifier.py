class ErrorQualifier:
    def qualify(self, wrong_word: str, suggestions: list[str]) -> bool:
        return True


class TsaErrorQualifier(ErrorQualifier):
    def qualify(self, wrong_word: str, suggestions: list[str]) -> bool:
        if not (wrong_word.endswith("тся") or wrong_word.endswith("ться")):
            return False

        return (wrong_word.replace("тся", "ться") in suggestions or
                wrong_word.replace("ться", "тся") in suggestions)
