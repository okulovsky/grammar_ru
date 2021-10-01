from .architecture import *
from yo_fluq_ds import *
from pathlib import Path
from ...common import DataBundle

class TikhonovDict:
    @staticmethod
    def read():
        return (Query
                .en(['test_Tikhonov.txt', 'train_Tikhonov.txt'])
                .select(lambda z: Path(__file__).parent/z)
                .select_many(Query.file.text)
                .select(lambda z: z.split('\t'))
                .select(lambda z: KeyValuePair(z[0], z[1]))
                .distinct(lambda z: z.key)
                )

    @staticmethod
    def read_as_df():
        result = []
        for kv in TikhonovDict.read():
            parts = kv.value.split('/')
            for i, part in enumerate(parts):
                ppart = part.split(':')
                result.append((kv.key, i, ppart[1], ppart[0]))
        return pd.DataFrame(result, columns=['word', 'morpheme_index', 'morpheme_type', 'morpheme'])


class MorphemeTikhonovEnricher(Enricher):
    def __init__(self, enabled_morphemes = None):
        self.tikhonov_df = TikhonovDict.read_as_df().set_index('word')
        if enabled_morphemes is not None:
            self.tikhonov_df = self.tikhonov_df.loc[self.tikhonov_df.morpheme_type.isin(enabled_morphemes)]

    def get_df_name(self) -> str:
        return 'tikhonov_morphemes'

    def enrich(self, db: DataBundle) -> None:
        df = db.src.word_id.to_frame()
        df = df.merge(db.pymorphy.normal_form.to_frame(), left_on='word_id', right_index=True)
        df = df.merge(self.tikhonov_df, left_on='normal_form', right_index=True)
        df = df.drop('normal_form',axis=1)
        df = df.set_index('word_id')
        db[self.get_df_name()] = df

