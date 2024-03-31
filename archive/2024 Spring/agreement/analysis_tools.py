import ast
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from tg.grammar_ru.common import Loc
from tg.grammar_ru.corpus import CorpusReader, CorpusBuilder, BucketCorpusBalancer
from tg.grammar_ru.corpus.corpus_reader import read_data
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Loc.root_path / 'environment.env')
from tg.grammar_ru.components.yandex_storage.s3_yandex_helpers import S3YandexHandler
from tg.grammar_ru.components.yandex_delivery.training_logs import S3TrainingLogsLoader, TrainingLogsViewer

from yo_fluq_ds import Queryable, Query, fluq
import plotly.express as px
from tg.grammar_ru.common import Separator

from typing import List, Union
import numpy as np
import torch
import math
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_tasks(bucket, tasks_list_s3_path):
    tmp_local_file = Loc.temp_path / tasks_list_s3_path.split('/')[-1]
    S3YandexHandler.download_file(bucket, tasks_list_s3_path, tmp_local_file)
    with open(tmp_local_file, 'r') as f:
        tasks = ast.literal_eval(f.read())
    return tasks


def plot_metrics(metrics, title=""):
    plt.plot(TrainingLogsViewer.get_metric_by_job(
        metrics, 'accuracy_display'), label='accuracy_display')
    plt.plot(TrainingLogsViewer.get_metric_by_job(
        metrics, 'accuracy_test'), label='accuracy_test')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_loss(metrics, title=""):
    plt.plot(TrainingLogsViewer.get_metric_by_job(
        metrics, 'loss'), label='loss')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_cm(cm):
    fig = go.Figure(data=go.Heatmap(z=cm,
                                    text=cm,
                                    x=cm.columns,
                                    y=cm.index,
                                    texttemplate="%{text}",
                                    colorscale='Blues'))
    fig.show()

def get_label(s):
    return int(s.split('_label_')[1])

def get_true_and_pred(result_df):
    pred_col_names = [c for c in result_df.columns if 'predicted_label' in c ]
    true_col_names = [c for c in result_df.columns if 'true_label' in c ]
    y_pred = result_df[pred_col_names].idxmax(axis="columns").apply(get_label)
    true_probs = result_df[true_col_names]
    y_true = true_probs.idxmax(axis="columns").apply(get_label)

    result_df['pred_label'] = y_pred
    result_df['true_label'] = y_true
    result_df['pred_score'] = result_df[pred_col_names].max(axis=1)

    return y_true, y_pred

def get_worst_words_sents(result_df, src, true_label: int, pred_label: int, worst_words_cnt: int):
    one_inst_another = result_df[(result_df.true_label == true_label) & (
        result_df.pred_label == pred_label)]
    thrsh = one_inst_another[f'predicted_label_{pred_label}'].sort_values(
        ascending=False).head(worst_words_cnt).min()
    worst_mistakes_scores = one_inst_another[
        one_inst_another[f'predicted_label_{pred_label}'] >= thrsh]

    worst_words = (src[src.word_id.isin(worst_mistakes_scores.word_id)]
                   [['word_id', 'sentence_id', 'word']])[:worst_words_cnt]
    worst_sents = worst_words['sentence_id'].unique()
    worst_sents_df = src[src.sentence_id.isin(worst_sents)]
    # worst_sents_df.loc[worst_sents_df.index, 'pred_score'] = -1
    # worst_sents_df.loc[worst_sents_df[worst_sents_df.word_id.isin(worst_mistakes_scores.word_id)].index, "pred_score"] = one_inst_another.pred_score.values
    return worst_words, worst_sents_df

def get_best_words_sents(result_df, src, pred_label: int, words_cnt: int):
    """ 
    Находит слова, в которых сеть была уверена в ответе и ответ верный
    """
    correct_df = result_df[result_df.true_label==pred_label]
    thrsh = correct_df[f'predicted_label_{pred_label}'].sort_values(
        ascending=False).head(words_cnt).min()
    best_scores = correct_df[correct_df[f'predicted_label_{pred_label}']>=thrsh]
    best_words = (src[src.word_id.isin(best_scores.word_id)])[['word_id', 'sentence_id', 'word']][:words_cnt]
    best_sents = best_words.sentence_id.unique()
    best_sents_df = src[src.sentence_id.isin(best_sents)]
    return best_words, best_sents_df

def get_training_results(bucket, job_name, project_name):
    tasks = get_tasks(bucket,job_name)

    loader = S3TrainingLogsLoader(bucket, project_name)
    metrics = loader.load_metrics(tasks)

    unzipped_folder = (Loc.root_path /
                    'temp'/'training_results' /
                    f'{tasks[0]}.unzipped')
    result_df = pd.read_parquet(unzipped_folder/'output'/'result_df.parquet')
    y_true, y_pred = get_true_and_pred(result_df)
    return metrics, result_df, y_true, y_pred, tasks
