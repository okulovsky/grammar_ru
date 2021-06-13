import pymorphy2
from yo_fluq_ds import *
import copy

class PyMorphyFeaturizer:
    def __init__(self):
        self.an = pymorphy2.MorphAnalyzer()
        sets = {
            'PARTS_OF_SPEECH': 'POS',
            'ANIMACY': 'animacy',
            'GENDERS': 'gender',
            'NUMBERS': 'number',
            'CASES': 'case',
            'ASPECTS': 'aspect',
            'TRANSITIVITY': 'transitivity',
            'PERSONS': 'person',
            'TENSES': 'tense',
            'MOODS': 'mood',
            'VOICES': 'voice',
            'INVOLVEMENT': 'involvement',
        }
        self.map = {}
        self.cat_order = list(sets.values())
        for cat_field, cat_name in sets.items():
            lst = list(getattr(self.an.TagClass, cat_field))
            for value in lst:
                if value in self.map:
                    raise (f'{value} is in key {self.map[v]} and in {cat_field}')
                self.map[value] = cat_name
        self.cache = {}

    def create_features(self, df):
        rows = []

        for src_row in Query.df(df):
            if src_row.word in self.cache:
                row = copy.copy(self.cache[src_row.word])
                row['word_id'] = src_row.word_id
                rows.append(row)
                continue
            row = {}
            row['word_id'] = src_row.word_id
            results = self.an.parse(src_row.word)
            row['alternatives'] = len(results)
            result = results[0]
            row['normal_form'] = result.normal_form
            row['score'] = result.score
            if len(results)>1:
                row['delta_score'] = results[0].score - results[1].score
            else:
                row['delta_score'] = row['score']
            for t in result.tag._grammemes_tuple:
                if t in self.map:
                    row[self.map[t]] = t
            rows.append(row)
            self.cache[src_row.word] = row

        pdf = pd.DataFrame(rows)
        columns = ['word_id','normal_form','alternatives','score','delta_score'] + self.cat_order
        for c in columns:
            if c not in pdf.columns:
                pdf[c]=None
        pdf = pdf[columns].set_index('word_id')
        return pdf