import numpy as np 
from googletrans import Translator
import matplotlib.pyplot as plt



def get_array_chapters(ru_retell_corpus):
    retell_author_df = ru_retell_corpus.get_toc() 
    retell = []
    for chapter in retell_author_df.index:
        chptr = ru_retell_corpus.get_bundles([chapter]).single().src
        sentences_id = np.array(chptr['sentence_id'].unique())
        sentences = [chptr['word'][chptr['sentence_id'] == sentence_id] for sentence_id in sentences_id]
        retell.append("\n".join(" ".join(sentence.values) for sentence in sentences))

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