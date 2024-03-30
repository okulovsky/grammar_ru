from yo_fluq_ds import *
from tg.grammar_ru import Separator
from tg.grammar_ru.features import SnowballFeaturizer

from tg.common import Loc
from tg.grammar_ru.algorithms.architecture import NlpAlgorithm
from tg.common.ml.batched_training import DataBundle, IndexedDataBundle


class AlternativeType(Enum):
    TSA = 1
    CHTOBY = 2
    NE = 3

class AlternativeAlgorithm(NlpAlgorithm):
    def __init__(self):
        super().__init__()

    def create_db(self, text: str):
        text_db = Separator.build_bundle(text, [SnowballFeaturizer()])
        # text_db.index = text_db.src[['sentence_id', 'word']]
        return text_db

    def create_index(self, db: DataBundle, alternative_type: AlternativeType ) -> pd.DataFrame:
        # Read the source data
        src_df = db.src

        # Filter words ending with 'тся' or 'ться'
        if alternative_type == AlternativeType.TSA:
            regex = r'.*ться$|.*тся$'
        elif alternative_type == AlternativeType.CHTOBY:
            regex = r'чтобы$|Чтобы$|что бы$|Что бы$'
        else:
            regex = r'не$|Нe$|ни$|Ни$'

        filtered_df = src_df[src_df['word'].str.contains(regex)]

        # Construct the index frame
        index_frame = filtered_df[['word_id', 'sentence_id']].copy()
        index_frame['label'] = 0
        index_frame['error'] = False
        index_frame['split'] = 'display'
        index_frame = index_frame.reset_index(drop=True)
        index_frame.index.name = 'sample_id'

        return index_frame

    def run(self, db: DataBundle, index: Optional[pd.Index] = None) -> pd.DataFrame:
        # for each enum_value in AlternativeType
        results = []
        for enum_value in AlternativeType:
            if enum_value != AlternativeType.TSA:
                break
            model_path = Loc.root_path / f'model/alternative_task_{str.lower(enum_value.name)}.pickle'
            idb = IndexedDataBundle(
                index_frame=self.create_index(db, enum_value),
                bundle=db
            )

            with open(model_path, 'rb') as handle:
                model = pickle.load(handle)

            prediction = model.predict(idb)
            result = pd.DataFrame()

            result['algorithm'] = 'alternative'
            result['error_type'] = NlpAlgorithm.ErrorTypes.Orthographic
            result['suggest'] = ...
            result['hint'] = ...

            for index, row in prediction.iterrows():
                word_id = row['word_id']
                word = db.src.loc[db.src['word_id'] == word_id, 'word'].iloc[0]
                error = prediction.at[index, 'predicted'] < 0.5
                result.at[index, 'error'] = error
                suggestion, hint = self._generate_suggestion_hint(word, error, enum_value)
                result.at[index, 'suggest'] = suggestion
                result.at[index, 'hint'] = hint

            results.append(result)

        return pd.concat(results)

    def _generate_suggestion_hint(self, word, error, alternative_type):
        # Logic to generate suggestions and hints based on the error
        if error:
            if alternative_type == AlternativeType.TSA:
                if 'тся' in word:
                    suggestion = word.replace('тся', 'ться')
                    hint = "Use 'ться' at the end of reflexive verbs in the infinitive form."
                else:
                    suggestion = word.replace('ться', 'тся')
                    hint = "Use 'тся' at the end of reflexive verbs in conjugated forms."

            elif alternative_type == AlternativeType.CHTOBY:
                if 'что бы' in word:
                    suggestion = word.replace('что бы', 'чтобы')
                    hint = "Use 'чтобы' as a single word to introduce subordinate clauses."
                else:
                    suggestion = word.replace('чтобы', 'что бы')
                    hint = "Use 'что бы' in situations where 'что' and 'бы' are separate parts of the sentence."

            elif alternative_type == AlternativeType.NE:
                if word.startswith('не'):
                    suggestion = word[2:]
                    hint = "Avoid using 'не' if the negation is not required."
                else:
                    suggestion = 'не' + word
                    hint = "Use 'не' for negation."

            else:
                suggestion = word
                hint = "Specific correction rule not found."
        else:
            suggestion = "No correction needed"
            hint = "The word appears to be correct."

        return suggestion, hint


    def _run_inner(self, db: DataBundle, index: pd.Index) -> Optional[pd.DataFrame]:
        pass
