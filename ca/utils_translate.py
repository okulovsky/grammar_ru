from typing import *
import numpy as np
from googletrans import Translator
import matplotlib.pyplot as plt
import re
from pathlib import Path
from grammar_ru.corpus import ParallelCorpus
from grammar_ru.corpus import CorpusBuilder
import os
from grammar_ru.corpus import CorpusReader
import shutil
from grammar_ru.common import Loc
import pandas as pd


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


def add_relation(df_1,df_2,name_1,name_2):
    rel_1 = pd.DataFrame({'file_1':df_1, 'file_2':df_2,'relation_name':f"{name_1}_{name_2}"})
    rel_2 = pd.DataFrame({'file_1':df_2, 'file_2':df_1,'relation_name':f"{name_2}_{name_1}"})
    rel = pd.concat([rel_1,rel_2])
    return rel

def translate_subcorpus(
        parallel_corpus_path: Path,
        subcorpus_to_translate_name: str,
        translated_subcorpus_name: str,
        custom_guid_factory: Optional[Callable[[int], str]] = None
):
    parallel_corpus = ParallelCorpus(parallel_corpus_path)
    subcorpus: CorpusReader = getattr(parallel_corpus, subcorpus_to_translate_name)
    text_subcorpus = get_array_chapters(subcorpus)
    trans = translate(text_subcorpus)
    result = ''

    for text in trans:
        result += "\n## part\n"

        result += text

    FOLDER = Loc.temp_path/'translate'
    shutil.rmtree(FOLDER, ignore_errors=True)
    os.makedirs(FOLDER, exist_ok=True)

    with open(FOLDER/'translate.md', 'w', encoding='utf-8') as file:
        file.write(result)

    CorpusBuilder.convert_interformat_folder_to_corpus(
        Loc.temp_path/'translate.base.zip',
        FOLDER,
        ['book'],
        custom_guid_factory=custom_guid_factory
    )

    reader = CorpusReader(Loc.temp_path/'translate.base.zip')

    CorpusBuilder.update_parallel_data(
        parallel_corpus_path,
        reader,
        translated_subcorpus_name,
        add_relation(subcorpus.get_toc().index, reader.get_toc().index, subcorpus_to_translate_name, translated_subcorpus_name)
    )








