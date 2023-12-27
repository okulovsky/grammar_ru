import numpy as np 
from googletrans import Translator
import matplotlib.pyplot as plt
import re
from pathlib import Path
from tg.grammar_ru.corpus import ParallelCorpus
from tg.grammar_ru.corpus import CorpusBuilder
import os
from tg.grammar_ru.corpus import CorpusReader
import shutil


def get_array_chapters(ru_retell_corpus):
    retell_author_df = ru_retell_corpus.get_toc() 
    retell = []
    for chapter in retell_author_df.index:
        chptr = ru_retell_corpus.get_bundles([chapter]).single().src
        sentences_id = np.array(chptr['sentence_id'].unique())


        sentences = [chptr['word'][chptr['sentence_id'] == sentence_id] for sentence_id in sentences_id]

        
        retell.append(re.sub(r'\s+(?=(?:[,.?!:;…]))', r'', "\n".join(" ".join(sentence.values) for sentence in sentences)))


    return retell



def translate(true_retell):
    translator = Translator()
    translations = translator.translate(true_retell, dest= 'ru')
    trans_retell = []
    for trans in translations:
        trans_retell.append(trans.text)

    return trans_retell 


def jac_metric(jac):
    fig, axis = plt.subplots(1, 1)
    axis.bar(range(len(jac)), jac)
    axis.set_title('Индекс Жаккара')


def add_dfs(name):
    frames = list(name.get_frames())
    dfs = dict(zip(name.get_toc().index,frames))

    return dfs


def translate_subcorpus(parallel_corpus_path: Path, name_subcorpus: str):
    parallel_corpus = ParallelCorpus(parallel_corpus_path)
    subcorpus = getattr(parallel_corpus, name_subcorpus)
    text_subcorpus = get_array_chapters(subcorpus)
    trans = translate(text_subcorpus)
    result = ''

    for text in trans:
        result += "\n## part\n"

        result += text

    os.makedirs('translate')

    with open(os.path.join('translate', 'translate.md'), 'w') as file:
        file.write(result)

    CorpusBuilder.convert_interformat_folder_to_corpus(
    Path('./files/translate.base.zip'),
    Path('./translate'),
    ['book'])

    reader = CorpusReader(Path('./files/translate.base.zip'))

    CorpusBuilder.update_parallel_data(
    parallel_corpus_path,
    add_dfs(reader),
    "ru_translate",
    None)
    shutil.rmtree("./translate")








