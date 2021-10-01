from unittest import TestCase
from grammar_ru.common import Separator, DataBundle
from grammar_ru.ml.features import PyMorphyFeaturizer
from yo_fluq_ds import *

text = '''
Шеститурбинный дрон медленно подлетел к зданию с неброской вывеской «SILICONIC, Inc» на крыше. На дрон были навьючены красные баллоны, в каких обычно перевозят газ. 
Дрон пробил окно, влетел внутрь — и вот из образовавшейся дыры вырвалось пламя, за ним — черный дым. Прямое включение прервало сериал, которым Патрик и Амелия наслаждались теплым летним днем. 
Через секунду в углу экрана возникло взбудораженное лицо новостного диктора. 
— Город под атакой. Напали на завод по производству микрочипов.
'''

pd.set_option('display.width',None)

class UpdateEnrichTestCase(TestCase):

    def check(self, df1, df2, column):
        self.assertListEqual(
            list(df1[column].isnull()),
            list(df2[column].isnull())
        )
        self.assertListEqual(
            list(df1.loc[~df1[column].isnull()][column]),
            list(df2.loc[~df2[column].isnull()][column]),
        )


    def test_update_enrich_pymorphy(self):
        enricher  = PyMorphyFeaturizer().as_enricher()

        pars0 = Query.en(text.split('\n')).where(lambda z: z!='').to_list()
        df0 = Separator.separate_paragraphs(pars0)
        db0 = DataBundle(src=df0)
        enricher.enrich(db0)

        pars1 = [
            pars0[1],
            pars0[0],
            'Вставленное предложение.',
            pars0[2]
        ]

        df_test = Separator.update_separation(df0,pars1, [1,0,None,2])
        db_test = DataBundle(src=df_test)
        enricher.update_enrich(db0, db_test)

        df_control = Separator.separate_paragraphs(pars1)
        db_control = DataBundle(src = df_control)
        enricher.enrich(db_control)

        py_test = db_test.pymorphy.reset_index().sort_values('word_id')
        py_control = db_control.pymorphy.reset_index().sort_values('word_id')

        for c in py_control.columns:
            self.check(py_control, py_test, c)



