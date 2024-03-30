import json
from pathlib import Path
from tg.grammar_ru import CorpusReader

def print_fragments(path: str):
    file = open(path)
    data = json.load(file)
    file.close()

    for fragment in data["fragments"]:
        input("next fragment")
        print(fragment['text'])

def print_frames(path: str):
    eng_corpus = path
    eng_corpus_reader = CorpusReader(eng_corpus)

    for frame in eng_corpus_reader.get_frames():
        input("next frame")
        print(frame)

if __name__ == '__main__':
    print_fragments()
