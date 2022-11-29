# from .architecture import *
import pymorphy2
from yo_fluq_ds import *
import copy

from tg.grammar_ru.ml.features import PyMorphyFeaturizer
from tg.grammar_ru.ml.features.architecture import *
from itertools import groupby


class GenderLabelPyMorphyFeaturizer(PyMorphyFeaturizer):

    def __init__(self):
        super(GenderLabelPyMorphyFeaturizer, self).__init__()

    def _featurize_inner(self, db: DataBundle):
        rows = []

        for src_row in Query.df(db.src).feed(fluq.with_progress_bar()):  # TODO delete progress bar
            if src_row.word in self.cache:
                row = copy.copy(self.cache[src_row.word])
                row['word_id'] = src_row.word_id
                rows.append(row)
                continue
            row = {f"gender_{gender}_score": 0 for gender in ['masc', 'femn', 'neut', 'None']}
            row['word_id'] = src_row.word_id
            results = self.an.parse(src_row.word)
            row['alternatives'] = len(results)
            for gender, scores in groupby([(t.score, t.tag.gender) for t in self.an.parse(src_row.word)],
                                          lambda x: x[1]):
                row[f'gender_{str(gender)}_score'] = sum(score_t[0] for score_t in scores)

            result = results[0]
            row['normal_form'] = result.normal_form
            row['score'] = result.score
            if len(results) > 1:
                row['delta_score'] = results[0].score - results[1].score
            else:
                row['delta_score'] = row['score']
            for t in result.tag._grammemes_tuple:
                if t in self.map:
                    row[self.map[t]] = t
            rows.append(row)
            self.cache[src_row.word] = row

        pdf = pd.DataFrame(rows)
        columns = ['word_id', 'normal_form', 'alternatives', 'score', 'delta_score',
                   'gender_masc_score', 'gender_femn_score','gender_neut_score', 'gender_None_score'] + self.cat_order
        for c in columns:
            if c not in pdf.columns:
                pdf[c] = None
        pdf = pdf[columns].set_index('word_id')
        pdf.POS = pdf.POS.fillna('NONE')
        return pdf
