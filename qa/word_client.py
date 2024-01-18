from typing import Type

import win32com.client

from enums.language import Language
from entities.spell_error import SpellError
from qa.entities.error_qualifier import ErrorQualifier, TrueErrorQualifier, TsaErrorQualifier


class MSWordClient:
    def __init__(self, language: Language) -> None:
        self.word = win32com.client.Dispatch('Word.Application')
        self.word.Visible = False

        self.document = self.word.Documents.Add()
        self.document.Range().LanguageID = language.value

    def spell_check(self, text: str, error_qualifier: Type[ErrorQualifier] = TrueErrorQualifier) -> list[SpellError]:
        self.document.Range().Text = text

        result = []

        for i in range(self.document.Words.Count):
            for error in self.document.Words.Item(i + 1).SpellingErrors:
                suggestions = [error.GetSpellingSuggestions().Item(i).Name
                               for i in range(1, error.GetSpellingSuggestions().Count + 1)]

                if error_qualifier.qualify(error.Text, suggestions):
                    result.append(SpellError(error.Text, i, error.Start, error.End, suggestions))

        return result

    def close(self) -> None:
        self.document.Close(SaveChanges=0)
        self.word.Quit()
