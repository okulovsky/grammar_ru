import json
from tg.grammar_ru import CorpusReader

def print_fragments():
    file = open("F:/grammar_ru/tg\projects/book_fragments/fragments/eng_cip_fragments.json")
    data = json.load(file)
    file.close()

    for fragment in data["fragments"]:
        input("next fragment")
        print(fragment['text'])

def print_frames():
    eng_corpus = "F:/grammar_ru/tg/projects/book_fragments/files/corpuses/corpus.zip"
    eng_corpus_reader = CorpusReader(eng_corpus)

    for frame in eng_corpus_reader.get_frames():
        input("next frame")
    

if __name__ == '__main__':
    print_fragments()
