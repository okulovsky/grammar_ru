import unittest
from diplom.utils.dialog_markuper import DialogMarkupFeaturizer
from grammar_ru.common import Separator, DataBundle


def _build_summary(df, markup):
    vdf = df.merge(markup, left_on='word_id', right_index=True).copy()
    vdf = vdf[['word', 'dialog_token_type']]
    vdf['next_dialog_token_type'] = vdf.dialog_token_type.shift(-1).fillna('')
    vdf['next_word'] = vdf.word.shift(-1).fillna('')
    vdf = vdf.loc[vdf.dialog_token_type != vdf.next_dialog_token_type]
    vdf = vdf[['word', 'next_word', 'dialog_token_type', 'next_dialog_token_type']]
    return [list(z[1]) for z in vdf.iterrows()]


def _build__diailog_type_summary(df, markup):
    vdf = df.merge(markup, left_on='word_id', right_index=True).copy()
    vdf = vdf[['word', 'dialog_type']]
    vdf['next_dialog_type'] = vdf.dialog_type.shift(-1).fillna('')
    vdf['next_word'] = vdf.word.shift(-1).fillna('')
    vdf = vdf.loc[vdf.dialog_type != vdf.next_dialog_type]
    vdf = vdf[['word', 'next_word', 'dialog_type', 'next_dialog_type']]
    return [list(z[1]) for z in vdf.iterrows()]

def _build__dialog_id_summary(df, markup):
    vdf = df.merge(markup, left_on='word_id', right_index=True).copy()
    vdf = vdf[['word', 'dialog_id']]
    vdf['next_dialog_id'] = vdf.dialog_id.shift(-1).fillna('')
    vdf['next_word'] = vdf.word.shift(-1).fillna('')
    vdf = vdf.loc[vdf.dialog_id != vdf.next_dialog_id]
    vdf = vdf[['word', 'next_word', 'dialog_id', 'next_dialog_id']]
    return [list(z[1]) for z in vdf.iterrows()]


def _get_summary(summarizer, s):
    df = Separator.separate_string(s)
    db = DataBundle(src=df)
    featurizer = DialogMarkupFeaturizer(['"'])
    featurizer.featurize(db)
    summary = summarizer(db.src, db.dialog_markup)
    return summary


class DialogMarkuperIntegrationTestCase(unittest.TestCase):
    def check_dialog_token_type(self, s, reference):
        summary = _get_summary(_build_summary, s)
        self.assertListEqual(reference, summary)

    def check_dialog_type(self, s, reference):
        summary = _get_summary(_build__diailog_type_summary, s)
        self.assertListEqual(reference, summary)

    def check_dialog_id(self, s, reference):
        summary = _get_summary(_build__dialog_id_summary, s)
        self.assertListEqual(reference, summary)

    def test_simple_dialog(self):
        s = '"Hello, how are you?"'
        self.check_dialog_token_type(s,
                                     [
                                         ['"', 'Hello', 'dialog-dash', 'speech'],
                                         ['Hello', ',', 'speech', 'dialog-symbol'],
                                         [',', 'how', 'dialog-symbol', 'speech'],
                                         ['you', '?', 'speech', 'dialog-symbol'],
                                         ['?', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', '', 'dialog-dash', '']
                                     ])
        self.check_dialog_type(s,
                               [
                                   ['"', '', 'dialog', ''],
                               ])

        self.check_dialog_id(s,[
            ['"', 'Hello', 1, 2],
            ['?', '"', 2, 3],
            ['"', '', 3, '']
        ])

    def test_incorrect_dialog_dash(self):
        s = 'John says: "Hello, how are you? and gone.\n "Have a nice day", he said.'
        self.check_dialog_token_type(s,
                                     [
                                         ['.', '"', 'none', 'dialog-dash'],
                                         ['"', 'Have', 'dialog-dash', 'speech'],
                                         ['day', '"', 'speech', 'dialog-dash'],
                                         ['"', ',', 'dialog-dash', 'action'],
                                         ['.', '', 'action', ''],
                                     ])
        self.check_dialog_type(s, [
            ['.', '"', "wrong", 'dialog'],
            ['.', '', 'dialog', ''],
        ])

        self.check_dialog_id(s,[
            ['.', '"', 1, 3],
            ['"', 'Have', 3, 4],
            ['day', '"', 4, 5],
            ['"', ',', 5, 6],
            ['.', '', 6, '']
        ])

    def test_simple_dialog_with_action(self):
        s = 'John says: "Hello, how are you?" and walk away.'
        self.check_dialog_token_type(s,
                                     [
                                         [':', '"', 'action', 'dialog-dash'],
                                         ['"', 'Hello', 'dialog-dash', 'speech'],
                                         ['Hello', ',', 'speech', 'dialog-symbol'],
                                         [',', 'how', 'dialog-symbol', 'speech'],
                                         ['you', '?', 'speech', 'dialog-symbol'],
                                         ['?', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', 'and', 'dialog-dash', 'action'],
                                         ['.', '', 'action', '']
                                     ])
        self.check_dialog_type(s,
                               [
                                   ['.', '', 'dialog', ''],
                               ])

        self.check_dialog_id(s,[
            [':', '"', 0, 1],
            ['"', 'Hello', 1, 2],
            ['?', '"', 2, 3],
            ['"', 'and', 3, 4],
            ['.', '', 4, '']
        ])

    def test_simple_dialog_with_action_and_text(self):
        s = 'It was a sunny day. John says: "Hello, how are you?" and walk away. Sun is shining.'
        self.check_dialog_token_type(
            s,
            [
                ['.', 'John', 'none', 'action'],
                [':', '"', 'action', 'dialog-dash'],
                ['"', 'Hello', 'dialog-dash', 'speech'],
                ['Hello', ',', 'speech', 'dialog-symbol'],
                [',', 'how', 'dialog-symbol', 'speech'],
                ['you', '?', 'speech', 'dialog-symbol'],
                ['?', '"', 'dialog-symbol', 'dialog-dash'],
                ['"', 'and', 'dialog-dash', 'action'],
                ['.', 'Sun', 'action', 'none'],
                ['.', '', 'none', '']
            ])
        self.check_dialog_type(s, [
            ['.', 'John', 'text', 'dialog'],
            ['.', 'Sun', 'dialog', 'text'],
            ['.', '', 'text', ''],
        ])

        self.check_dialog_id(s,[
            [':', '"', 0, 1],
            ['"', 'Hello', 1, 2],
            ['?', '"', 2, 3],
            ['"', 'and', 3, 4],
            ['.', '', 4, '']
        ])

    def test_many_dialogs_in_1_sentence(self):
        s = '"The Sun is shining," he looks at the sky and continue: "good weather to go."'
        self.check_dialog_token_type(s,
                                     [
                                         ['"', 'The', 'dialog-dash', 'speech'],
                                         ['shining', ',', 'speech', 'dialog-symbol'],
                                         [',', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', 'he', 'dialog-dash', 'action'],
                                         [':', '"', 'action', 'dialog-dash'],
                                         ['"', 'good', 'dialog-dash', 'speech'],
                                         ['go', '.', 'speech', 'dialog-symbol'],
                                         ['.', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', '', 'dialog-dash', '']
                                     ])
        self.check_dialog_type(s,
                               [
                                   ['"', '', 'dialog', ''],
                               ])

        self.check_dialog_id(s,[
            ['"', 'The', 1, 2],
            [',', '"', 2, 3],
            ['"', 'he', 3, 4],
            [':', '"', 4, 5],
            ['"', 'good', 5, 6],
            ['.', '"', 6, 7],
            ['"', '', 7, '']
        ])

    def test_1_dialog_in_many_sentences(self):
        s = '"I need to go. Have a good day," he said.'
        self.check_dialog_token_type(s,
                                     [
                                         ['"', 'I', 'dialog-dash', 'speech'],
                                         ['go', '.', 'speech', 'dialog-symbol'],
                                         ['.', 'Have', 'dialog-symbol', 'speech'],
                                         ['day', ',', 'speech', 'dialog-symbol'],
                                         [',', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', 'he', 'dialog-dash', 'action'],
                                         ['.', '', 'action', '']
                                     ])
        self.check_dialog_type(s,
                               [
                                   ['.', '', 'dialog', ''],
                               ])
        self.check_dialog_id(s,[
            ['"', 'I', 1, 2],
            [',', '"', 2, 3],
            ['"', 'he', 3, 4],
            ['.', '', 4, '']
        ])

    def test_quotation_inside_dialog_incorrect_behavior(self):
        s = """"I need to go to "West Bank" immediately." he said."""
        self.check_dialog_token_type(s,
                                     [
                                      ['"', 'I', 'dialog-dash', 'speech'],
                                      ['to', '"', 'speech', 'dialog-dash'],
                                      ['"', 'West', 'dialog-dash', 'action'],
                                      ['Bank', '"', 'action', 'dialog-dash'],
                                      ['"', 'immediately', 'dialog-dash', 'speech'],
                                      ['immediately', '.', 'speech', 'dialog-symbol'],
                                      ['.', '"', 'dialog-symbol', 'dialog-dash'],
                                      ['"', 'he', 'dialog-dash', 'action'],
                                      ['.', '', 'action', '']
                                     ])
        self.check_dialog_type(s,
                               [
                                   ['.', '', 'dialog', ''],
                               ])
        self.check_dialog_id(s,
                             [
            ['"', 'I', 1, 2],
            ['to', '"', 2, 3],
            ['"', 'West', 3, 4],
            ['Bank', '"', 4, 5],
            ['"', 'immediately', 5, 6],
            ['.', '"', 6, 7],
            ['"', 'he', 7, 8],
            ['.', '', 8, '']
        ])

    def test_hard_text_with_all_stuff(self):
        s = """It was a sunny day. John says: "Hello, how are you?". 
        "The Sun is shining," he looks at the sky and continue: "good weather to go." 
        "I need to go. Have a good day," he said and walk away. His task is done."""
        self.check_dialog_token_type(s,
                                     [
                                         ['.', 'John', 'none', 'action'],
                                         [':', '"', 'action', 'dialog-dash'],
                                         ['"', 'Hello', 'dialog-dash', 'speech'],
                                         ['Hello', ',', 'speech', 'dialog-symbol'],
                                         [',', 'how', 'dialog-symbol', 'speech'],
                                         ['you', '?', 'speech', 'dialog-symbol'],
                                         ['?', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', '.', 'dialog-dash', 'action'],
                                         ['.', '"', 'action', 'dialog-dash'],
                                         ['"', 'The', 'dialog-dash', 'speech'],
                                         ['shining', ',', 'speech', 'dialog-symbol'],
                                         [',', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', 'he', 'dialog-dash', 'action'],
                                         [':', '"', 'action', 'dialog-dash'],
                                         ['"', 'good', 'dialog-dash', 'speech'],
                                         ['go', '.', 'speech', 'dialog-symbol'],
                                         ['.', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', 'I', 'dialog-dash', 'speech'],
                                         ['go', '.', 'speech', 'dialog-symbol'],
                                         ['.', 'Have', 'dialog-symbol', 'speech'],
                                         ['day', ',', 'speech', 'dialog-symbol'],
                                         [',', '"', 'dialog-symbol', 'dialog-dash'],
                                         ['"', 'he', 'dialog-dash', 'action'],
                                         ['.', 'His', 'action', 'none'],
                                         ['.', '', 'none', ''],
                                     ])
        self.check_dialog_type(s, [
            ['.', 'John', 'text', 'dialog'],
            ['.', 'His', 'dialog', 'text'],
            ['.', '', 'text', ''],
        ])
        s = """It was a sunny day. John says: "Hello, how are you?". 
        "The Sun is shining," he looks at the sky and continue: "good weather to go." 
        "I need to go. Have a good day," he said and walk away. His task is done."""
        self.check_dialog_id(s, [
            [':', '"', 0, 1],
            ['"', 'Hello', 1, 2],
            ['?', '"', 2, 3],
            ['"', '.', 3, 4],
            ['.', '"', 4, 5],
            ['"', 'The', 5, 6],
            [',', '"', 6, 7],
            ['"', 'he', 7, 8],
            [':', '"', 8, 9],
            ['"', 'good', 9, 10],
            ['.', '"', 10, 11],
            ['"', '"', 11, 13],
            ['"', 'I', 13, 14],
            [',', '"', 14, 15],
            ['"', 'he', 15, 16],
            ['.', '', 16, ''],
        ])
