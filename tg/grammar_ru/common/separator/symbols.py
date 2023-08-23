import string

import homoglyphs as hg

homoglyphs = hg.Homoglyphs(categories=('LATIN', 'COMMON'))


class Symbols:
    RUSSIAN_LETTERS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    RUSSIAN_WORD_SYMBOLS = RUSSIAN_LETTERS + '-' + chr(8242)
    PUNCTUATION = ',.–?!—…:«»";()“”„-'
    APOSTROPHS = "'" + chr(8217)
    PUNCTUATION_OR_SPACE = PUNCTUATION + ' \n\t'
    EN_LETTERS = string.ascii_letters
    EN_WORD_SYMBOLS = EN_LETTERS + '-' + '\'' + ''.join(homoglyphs.get_combinations('\''))
    EN_PUNCT = ''.join(set(PUNCTUATION + ''.join(homoglyphs.get_combinations('-')) +
                           "«»‘’‚‛“”„‟‹›❛❜❝❞❮❯〝〞〟＂❟❠⹂🙶🙷🙸＇") - {'⁃', '﹘'})
