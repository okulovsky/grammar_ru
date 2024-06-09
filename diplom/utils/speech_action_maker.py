from grammar_ru.features.architecture import *
from collections import defaultdict
import pandas as pd
import spacy
from diplom.utils.Embeders import SpacyEmbeder
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import torch
import numpy as np

nlp = spacy.load('en_core_web_sm')

# TODO add more verb to collect
with open('../verbs.txt') as f:
    verbs = [line.rstrip() for line in f]
embeder = SpacyEmbeder()
verbs_emded = torch.stack([embeder.get_embedding(token) for token in verbs])


def make_merged_df(bundle, only_dialog=False):
    src, dialog_markup = bundle.src, bundle.dialog_markup
    merged = dialog_markup.merge(src, on='word_id')
    return merged[(merged.dialog_token_type != 'none')] if only_dialog else merged


def make_dialog_df(df, diaog_info_columns):
    dialogs = df.loc[df.dialog_type == 'dialog'][diaog_info_columns]  # .rename(columns={'word_y': 'word'})
    dialogs['word'] = dialogs.word + dialogs.assign(space=' ').space * dialogs.word_tail
    t_dialog = dialogs.groupby('dialog_id').word.sum()
    text_dialog = df[['sentence_id', 'dialog_id']].merge(t_dialog, on='dialog_id')[
        ['word', 'dialog_id']].drop_duplicates()
    return text_dialog


def constuct_speech_action(text_dialog, dialog_seps):
    in_dialog = False
    speech, action = None, None
    sample_id = 0
    ans_dict = defaultdict(list)
    for row in text_dialog.itertuples():
        _, word, dialog_id = row
        if in_dialog:
            if word.strip() in dialog_seps:
                in_dialog = False
            else:
                speech = word
        else:
            if word in dialog_seps:  # Т.е. если сначала был экшен, а потом спич, то мы сотрём экщен, тем самым получаем только экшены после спича.
                in_dialog = True
                speech, action = None, None
            else:
                action = word
        if speech is not None and action is not None:  # Такая ситуация возможна только тогда, когда мы не в диалоге, т.к. когда диалог открывается, спич и экшн обнуляются.
            # TODO Добавить фильтр на длину action/ обрезать экшн по длине и если там нет синонима said, то выкидывать и тд
            ans_dict["sample_id"].append(sample_id)
            ans_dict["speech"].append(speech)
            ans_dict["action"].append(action)
            speech, action = None, None
            sample_id += 1
    return pd.DataFrame.from_dict(ans_dict)


def take_only_verb_from_action(speech_action, cos_sims_d, threshold=0.8):
    new_ans = defaultdict(list)
    sample_id = 0
    for row in speech_action.itertuples():
        speech, action = row.speech, row.action
        doc = nlp(action)
        verb_only = [token for token in doc if token.pos_ == 'VERB']
        if len(verb_only) == 0:
            continue
        lemma_embed, action_verb_word = zip(*[(token.lemma_, token.orth_) for token in verb_only])
        maxis = []
        for token in lemma_embed:
            if token not in cos_sims_d:
                emb = embeder.get_embedding(token)
                cos_sims_d[token] = np.max(cos_sim(torch.unsqueeze(emb, 0), verbs_emded))
            maxis.append(cos_sims_d[token])
        amax = np.argmax(maxis)
        if maxis[amax] > threshold:
            new_ans["sample_id"].append(sample_id)
            sample_id += 1
            new_ans["speech"].append(speech)
            new_ans["action"].append(action_verb_word[amax])

    return pd.DataFrame.from_dict(new_ans)


class SpeechActionFeaturizer(SimpleFeaturizer):
    def __init__(self, dialog_seps):
        super(SpeechActionFeaturizer, self).__init__('speech_action', False)
        self.diaog_info_columns = ['word', 'dialog_token_type', 'word_id', 'dialog_id', 'sentence_id',
                                   'paragraph_id', 'word_tail']
        self.dialog_seps = dialog_seps
        self.cos_sims_d = dict()

    def _featurize_inner(self, db: DataBundle):
        df = make_merged_df(db)
        text_dialog = make_dialog_df(df, self.diaog_info_columns)
        speech_action = constuct_speech_action(text_dialog, self.dialog_seps)
        only_verbs_in_action = take_only_verb_from_action(speech_action, self.cos_sims_d)
        return only_verbs_in_action