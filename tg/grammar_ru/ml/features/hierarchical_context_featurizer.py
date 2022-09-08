from typing import *
from .architecture import *
import pandas as pd
from ....common import DataBundle
from yo_fluq_ds import Query, fluq



def build_basic_df(bundle: DataBundle):
    df = bundle.slovnet[['syntax_parent_id', 'relation']]
    df = df.merge(bundle.src.set_index('word_id')[['sentence_id', 'word']], left_index=True, right_index=True)
    df['root'] = 'No'
    df.loc[df.syntax_parent_id == -1, 'root'] = 'Good'
    return df


def add_lints_to(df):
    ldf = df.loc[df.syntax_parent_id != -1].groupby('syntax_parent_id').size().to_frame('links_to')
    ldf = df[[]].merge(ldf, left_index=True, right_index=True, how='left')
    ldf = ldf.fillna(0)
    df['links_to'] = ldf.links_to
    df['cycle_status'] = 'No'


def pick_roots_for_rootless_sentences(df):
    qdf = df.assign(is_root = (df.syntax_parent_id==-1))
    no_roots = qdf.groupby('sentence_id').is_root.any().feed(lambda z: z.loc[~z]).index
    qdf = qdf.loc[qdf.sentence_id.isin(no_roots)]
    qdf = qdf.feed(fluq.add_ordering_column('sentence_id',('links_to', False)))
    fix_index = qdf.loc[qdf.order==0].index
    df.loc[fix_index,'syntax_parent_id'] = -1
    df.loc[fix_index,'root'] = 'Picked'


def compute_multiple_roots(df):
    rdf = df.loc[df.syntax_parent_id==-1].groupby('sentence_id').size().to_frame('roots')
    mult_root = df.loc[df.sentence_id.isin(rdf.loc[rdf.roots>1].index)]
    mult_root = mult_root.loc[mult_root.syntax_parent_id==-1]
    mult_root = mult_root.feed(fluq.add_ordering_column('sentence_id',('links_to',False), 'order'))
    good_roots = mult_root.loc[mult_root.order==0].index
    bad_roots = mult_root.loc[mult_root.order>0].index
    df.loc[good_roots,'root'] = 'KeptMultRoot'
    df.loc[bad_roots,'root'] = 'DeletedMultRoot'
    return rdf


def fix_roots(df):
    df['is_root'] = df.root.isin(['Good', 'KeptMultRoot', 'Picked'])
    root_count = df.groupby('sentence_id').is_root.sum()
    missing_sentences = df.loc[~df.sentence_id.isin(root_count.index)]
    if missing_sentences.shape[0] > 0:
        raise ValueError(f'Sentences missing root: {missing_sentences.sentence_id.unique()}')
    if root_count.loc[root_count > 1].shape[0] > 0:
        raise ValueError(f"Sentences with too much roots: {root_count.loc[root_count > 1].index})")

    rdf = df.loc[df.is_root][['sentence_id', 'root']]
    rdf = rdf.reset_index().rename(columns={'word_id': 'correct_root'}).set_index('sentence_id')
    rdf = df[['sentence_id']].merge(rdf, left_on='sentence_id', right_index=True, how='left')
    df['correct_root'] = rdf.correct_root

    if df.correct_root.isnull().any():
        raise ValueError(f"Correct_root is not set for rows {df.loc[df.correct_root.isnull()].index}")
    df.loc[df.root == 'DeletedMultRoot', 'syntax_parent_id'] = df.loc[df.root == 'DeletedMultRoot', 'correct_root']
    if not (df.is_root == (df.syntax_parent_id == -1)).all():
        raise ValueError(
            f'Something is still wrong with the roots {df.loc[df.is_root != (df.syntax_parent_id == -1)].index}')


def build_closure(reldf, limit):
    reldf = reldf[['syntax_parent_id', 'relation']]
    tdf = reldf
    relation = []
    for step in range(limit):
        tdf = tdf.loc[tdf.syntax_parent_id >= 0]
        if tdf.shape[0] == 0:
            break
        rel_part = tdf.assign(distance=step + 1).reset_index()
        relation.append(rel_part)
        tdf = tdf.drop('relation', axis=1)
        tdf = tdf.rename(columns={'syntax_parent_id': 'current'}).merge(reldf, left_on='current',
                                                                        right_index=True).drop('current', axis=1)
    ddf = pd.concat(relation)
    ddf.index = list(range(ddf.shape[0]))
    ddf.index.name = 'entry_id'
    return ddf


def check_closure(df):
    cdf = build_closure(df, 100)
    in_cycle = cdf[cdf.word_id == cdf.syntax_parent_id].word_id.unique()
    df.loc[in_cycle, 'cycle_status'] = 'Yes'
    df['in_cycle'] = False
    df.loc[in_cycle, 'in_cycle'] = True
    bad_sentences = df.loc[in_cycle].sentence_id.unique()
    good_df = df.loc[~df.sentence_id.isin(bad_sentences)]
    good_closures = cdf.loc[cdf.word_id.isin(good_df.index)]
    bad_df = df.loc[df.sentence_id.isin(bad_sentences)]
    return good_df, good_closures, bad_df


def break_cycles(df):
    df = df.copy()
    has_cycles = df.groupby('sentence_id').in_cycle.any()
    if not has_cycles.all():
        raise ValueError(f'some sentences do not have cycles, {has_cycles.loc[~has_cycles]}')
    xdf = df.loc[df.in_cycle].feed(fluq.add_ordering_column('sentence_id', ('links_to', False), 'breaking_order'))
    fixes = xdf.loc[xdf.breaking_order==0].index
    df.loc[fixes,'syntax_parent_id'] = df.loc[fixes, 'correct_root']
    df.loc[fixes, 'cycle_status'] = 'Broken'
    return df


def execute_algorithm(db: DataBundle, N=30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = build_basic_df(db)
        add_lints_to(df)
        pick_roots_for_rootless_sentences(df)
        compute_multiple_roots(df)
        fix_roots(df)

        good_dfs = []
        closures = []
        tdf = df

        for i in range(N):
            if tdf.shape[0]==0:
                return pd.concat(good_dfs), pd.concat(closures)
            good, closure, tdf = check_closure(tdf)
            good_dfs.append(good)
            closures.append(closure)
            tdf = break_cycles(tdf)

        raise ValueError(f'Cycles were not eliminated in {N} iterations')

def _validate_set(set1, set2, description):
    if len(set(set1)-set(set2))!=0:
        raise ValueError(f'{description}\n{set(set1)-set(set2)}')


def validate_result(src, sdf, cdf):
    _validate_set(src.word_id, sdf.index, 'Some words from source are not found in fixed syntax')
    _validate_set(sdf.index, src.word_id, 'Some words from fixed syntax are not in the original source')
    cl = set(cdf.word_id).union(cdf.syntax_parent_id)
    _validate_set(cl, src.word_id, 'Some words from closure came from nowhere')
    bad_sentences = src.groupby('sentence_id').size().feed(lambda z: z.loc[z==1]).index
    due_words = src.loc[~src.sentence_id.isin(bad_sentences)].word_id
    _validate_set(due_words, cl, "Some words are not participating in closure")



class SyntaxTreeFeaturizer(Featurizer):
    def __init__(self, add_fixes = True, add_closures = True):
        self.add_fixes = add_fixes
        self.add_closures = add_closures

    def get_frame_names(self) -> List[str]:
        return ['syntax_fixes','syntax_closure']

    def featurize(self, db: DataBundle) -> None:
        sdf, cdf = execute_algorithm(db)
        validate_result(db.src, sdf, cdf)
        if self.add_fixes:
            db['syntax_fixes'] = sdf.drop(['sentence_id','relation','word','is_root','in_cycle','links_to'],axis=1)
        if self.add_closures:
            db['syntax_closure'] = cdf

