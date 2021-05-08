from yo_fluq_ds import *
from pathlib import Path


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
