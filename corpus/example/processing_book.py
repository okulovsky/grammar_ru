from tg.grammar_ru.ml.corpus.formats import Fb2Scripts
from pathlib import Path

if __name__ == '__main__':
    src = Path(__file__).parent/'raw/book.zip'
    dst = Path(__file__).parent/'processed/book.md'
    Fb2Scripts.convert_file(src, src.parent, dst.parent)
