import win32com.client
from enums.language import Language
from entities.spell_error import SpellError


class MSWordClient:
    def __init__(self, language: Language) -> None:
        self.word = win32com.client.Dispatch('Word.Application')
        self.word.Visible = False

        self.document = self.word.Documents.Add()
        self.document.Range().LanguageID = language.value

    def spell_check(self, text: str) -> list[SpellError]:
        self.document.Range().Text = text

        result = []

        # for i in self.document.Words:
        #     for j in i.SpellingErrors:
        #         print(j.Text)

        for rng in self.document.Range().SpellingErrors:
            suggestions = [rng.GetSpellingSuggestions().Item(i).Name
                           for i in range(1, rng.GetSpellingSuggestions().Count + 1)]

            result.append(SpellError(rng.Text, text, suggestions))

        return result

    def close(self) -> None:
        self.document.Close(SaveChanges=0)
        self.word.Quit()


ms_word = MSWordClient(Language.RUSSIAN)
print(ms_word.spell_check("Кот любит купатся в море. Кот любит дратся в море"))
