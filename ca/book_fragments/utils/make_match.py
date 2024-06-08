import pandas as pd

def make_match(book_chapter_df, retell_chapter_df, matcher):
    matched_ids, matched_sentences = matcher.get_matches(book_chapter_df, retell_chapter_df, need_matching_df=True)
    matched_ids['MatchedWithSentence'] = pd.Series(matched_sentences.array)
    matched_ids = matched_ids.dropna()


    df_to_display = pd.merge(book_chapter_df, matched_ids, left_on='sentence_id', right_on='sentence_id', how='left')
    sentences_to_display = matcher.viewer.to_sentences_strings(df_to_display, 'MatchedWith').to_frame()
    sentences_to_display = pd.merge(df_to_display, sentences_to_display, left_on='MatchedWith', right_on='MatchedWith', how='left')


    sent_dataset = sentences_to_display[['MatchedWithSentence', 'word_print']]
    sent_dataset = sent_dataset.drop_duplicates(subset=['word_print'], keep='first', )
    sent_dataset = sent_dataset.reset_index(drop=True)

    return sent_dataset.to_json(orient="records", force_ascii=False)