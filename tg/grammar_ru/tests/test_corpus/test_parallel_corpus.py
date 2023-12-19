from pandas import DataFrame

from tg.grammar_ru.corpus import CorpusReader, CorpusBuilder, ParallelCorpus
from tg.grammar_ru.corpus.formats import InterFormatParser
from uuid import uuid4
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from unittest import TestCase
from pathlib import Path
import os

global_sub_names = ['a', 'b', 'c']


def get_dfs_and_relations_and_uids() -> Tuple[List[Dict[str, pd.DataFrame]], DataFrame, List[List[str]]]:
    sub_corpuses = [[i for i in InterFormatParser(Path('/test'), Path(f'/test/{letter}'), ['folder', 'name'],
                                                  mock=text).parse().to_list()]
                    for text, letter in zip([text_1, text_2, text_3], ['a/b.md', 'b/c.md', 'd/e.md'])]
    sub_uids = [[str(uuid4()) for _ in range(2)] for __ in range(3)]
    sub_names = global_sub_names
    relations = []
    for sub_name_1, sub_uids_1 in zip(sub_names, sub_uids):
        for sub_name_2, sub_uids_2 in zip(sub_names, sub_uids):
            if sub_name_1 == sub_name_2:
                continue
            relations.append(pd.DataFrame(
                {'file_1': sub_uids_1, 'file_2': sub_uids_2, 'relation_name': f'{sub_name_1}-{sub_name_2}'}))
    relations = pd.concat(relations)
    dfs = [{k: v.df for k, v in zip(sub_uids, sub_corpus)} for sub_uids, sub_corpus in
           zip(sub_uids, sub_corpuses)]
    return dfs, relations, sub_uids


def write_data_to_coprus(dfs, relations, corpus_file, none_relation_mode=False):
    if os.path.exists(corpus_file):
        os.remove(corpus_file)
    subcorpus_col_name = "relation_name"
    f_s_relations = relations.loc[relations.relation_name.isin(['a-b', 'b-a'])] if not none_relation_mode else None
    added_relations = relations.loc[~relations.relation_name.isin(['a-b', 'b-a'])] if not none_relation_mode else None
    builder = CorpusBuilder()

    builder.update_parallel_data(corpus_file, dfs[0], global_sub_names[0], None, subcorpus_col_name)

    builder.update_parallel_data(corpus_file, dfs[1], global_sub_names[1], f_s_relations, subcorpus_col_name)

    builder.update_parallel_data(corpus_file, dfs[2], global_sub_names[2], added_relations, subcorpus_col_name)


class ParallelCorpusTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dfs, relations, uids = get_dfs_and_relations_and_uids()
        self.dfs = dfs
        self.relations = relations
        self.uids = uids
        self.corpus_file = Path(__file__).parent / 'temp_corpus.zip'

    def test_buider_update_parallel_data(self):
        dfs, relations = self.dfs, self.relations
        try:
            write_data_to_coprus(dfs, relations, self.corpus_file)
        except Exception:
            raise Exception("Incorrect update_parallel_data work. Other tests may not work properly.")
        os.remove(self.corpus_file)

    def test_unique_word_ids(self):
        dfs, relations, uids, corpus_file = self.dfs, self.relations, self.uids, self.corpus_file
        write_data_to_coprus(dfs, relations, corpus_file)
        reader = CorpusReader(corpus_file)
        all_frames = pd.concat(list(reader.get_frames()))
        assert (all_frames.word_id.unique() == all_frames.word_id.values).all()
        # нет теста на уникальность предложений между файлами, т.к. файл может быть слишком большой - разбился на два файла, но номер предложения один и тот же. Это не проверить.
        os.remove(corpus_file)

    def test_get_relation(self):
        required_columns = ('file_1', 'file_2', 'relation_name')
        dfs, relations, uids, corpus_file = self.dfs, self.relations, self.uids, self.corpus_file
        write_data_to_coprus(dfs, relations, corpus_file)
        reader = CorpusReader(corpus_file)
        relation = reader.get_relations()
        assert all(required_column in relation.columns for required_column in required_columns)
        assert len(relation) == len(uids) * len(uids[0]) * 2
        os.remove(corpus_file)

        write_data_to_coprus(dfs, None, corpus_file, none_relation_mode=True)
        try:
            relation = CorpusReader(corpus_file).get_relations()
            assert len(relation) == 0
            assert all(required_column in relation.columns for required_column in required_columns)
        except:
            raise Exception('Zero size or None relation doesnt work correctly')

    def test_get_src(self):
        dfs, relations, uids, corpus_file = self.dfs, self.relations, self.uids, self.corpus_file
        write_data_to_coprus(dfs, relations, corpus_file)
        sub_uids = [[sub_uids[i] for sub_uids in uids] for i in range(len(uids[0]))]
        sub_names = ['a', 'b', 'c']
        dict_uids = [{name: uid for name, uid in zip(sub_names, sub_uid)} for sub_uid in sub_uids]

        reader = CorpusReader(corpus_file)
        fragments = list(reader.get_src(uids))
        dict_fragments = list(reader.get_src(dict_uids))
        assert len(fragments) == len(uids)
        assert isinstance(fragments[0], list)
        assert all(len(fragment) == len(uids[0]) for fragment in fragments)
        assert len(dict_fragments) == len(uids[0])
        assert isinstance(dict_fragments[0], dict)
        assert all(len(dict_fragment) == len(uids) for dict_fragment in dict_fragments)
        os.remove(corpus_file)

    def test_sub_corpus_dot_expression(self):
        dfs, relations, uids, corpus_file = self.dfs, self.relations, np.array(self.uids), self.corpus_file
        write_data_to_coprus(dfs, relations, corpus_file)
        parallel_corpus = ParallelCorpus(corpus_file, 'relation_name')
        ids = list(range(len(global_sub_names)))
        for i in ids:
            sub_corpus_name = global_sub_names[i]
            other_sub_ids = np.delete(ids, i)
            incorrect_uids = uids[other_sub_ids]
            correct_uids = [list(uids[i])]
            reader: CorpusReader = getattr(parallel_corpus, sub_corpus_name)
            try:
                corerct_frames = list(reader.get_src(correct_uids))
            except (TypeError or ValueError):
                raise Exception('Cannot find correct uids. Trouble with gettatr in ParallelCorpus')

            try:
                incorerct_frames = list(reader.get_src(incorrect_uids))
            except (TypeError or ValueError):
                pass
            else:
                raise Exception('Find incorrect uids. Trouble with gettatr in ParallelCorpus')

            example: CorpusReader = parallel_corpus.a
            example_toc = example.get_toc()

        incorerct_sub_corpus_type = 'incorerct_type'
        try:
            getattr(parallel_corpus, incorerct_sub_corpus_type)
        except AttributeError:
            pass
        else:
            raise Exception(
                "Parallel corpus didn't raise an AttributeError on non- existent sub_corpus type. Trouble with gettatr in ParallelCorpus  ")
        os.remove(corpus_file)

    def test_get_mapped_data(self):
        dfs, relations, uids, corpus_file = self.dfs, self.relations, self.uids, self.corpus_file
        write_data_to_coprus(dfs, relations, corpus_file)
        parallel_corpus = ParallelCorpus(corpus_file, 'relation_name')
        relation_names = relations.relation_name.unique()
        for name in relation_names:
            my_relation = relations.loc[relations.relation_name == name]
            relations_names_to_map = relations.loc[
                relations.relation_name.str.startswith(name[0]), "relation_name"].unique()
            file_names = my_relation.file_1.values
            mapped_data = list(parallel_corpus.get_mapped_data(file_names, relations_names_to_map))
            for mapped_dict, correct_file_id in zip(mapped_data, file_names):
                for mapped_relation_name, mapped_frame in mapped_dict.items():
                    mapped_file_id = mapped_frame.file_id.unique()[0]
                    mapped_source_id = relations.loc[(relations.file_2 == mapped_file_id) & (
                            relations.relation_name == mapped_relation_name), 'file_1'].values[0]
                    assert mapped_source_id == correct_file_id
        os.remove(corpus_file)


text_1 = '''
# h1
# h1.1
## h2
Первый параграф. 

Второй.

## h22

Третий.
'''

text_2 = '''
# Первая строка
# Вторая строка
## Третья строка
Первый параграф. 

Второй.

## Четвертая строка

Третий.
'''

text_3 = '''
# Первое начало
# Второе начало
## Третье начало
Первый параграф. 

Второй.

## Четвертое начало

Третий.
'''
