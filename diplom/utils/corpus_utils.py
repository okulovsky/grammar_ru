import numpy as np
from typing import Any, Tuple
from typing import List
from grammar_ru.common import Separator


class CorpusFramework:
    def __init__(self, corpus):
        self.corpus = corpus
        self.viewer = Separator.Viewer()

    def get_books_by_author(self, author_name: str) -> List[Any]:
        """
        Take author name and return tuple of: books_info, retell_books_info, summary_books_info
        """
        author_df = self.corpus.get_toc()
        author_df = author_df[author_df['author'] == author_name]
        author_books = [author_df[author_df['book_name'] == name_of_book] for name_of_book in
                        author_df['book_name'].unique()]
        return author_books

    def __get_sentences_ids_chapter(self, chapter) -> Tuple[List[Any], List[Any], Any]:
        corpus = self.corpus
        chptr = corpus.get_bundles([chapter]).single()
        chapter_df = chptr.src
        sentences_id = np.array(chapter_df['sentence_id'].unique())
        sentences_ids = [chapter_df['word_id'][chapter_df['sentence_id'] == sentence_id]
                         for sentence_id in sentences_id]
        sentences = [chapter_df['word'].values[chapter_df['sentence_id'] == sentence_id]
                     for sentence_id in sentences_id]
        return sentences, sentences_ids, chptr

    def __get_chapter_as_df_by_id(self, chapter_id: str, ):
        corpus = self.corpus
        chptr = corpus.get_bundles([chapter_id]).single()
        chapter_df = chptr.src
        return chapter_df

    def get_chapter_as_df(self, text, chapter, type='src'):
        chptr = self.get_chapter_as_bundle(text, chapter)
        chapter_df = chptr[type]
        return chapter_df

    def get_chapter_as_bundle(self, text, chapter):
        corpus = self.corpus
        if isinstance(chapter, int):
            chapter_id = text.index[chapter]
        elif isinstance(chapter, str):
            chapter_id = chapter
        else:
            raise ValueError("Chapter must be a string or integer")
        chptr = corpus.get_bundles([chapter_id]).single()
        return chptr

    def show_as_text(self, df, as_html=False):
        return self.viewer.to_html_display(df) if as_html else self.viewer.to_text(df)

    def get_book_by_author_and_series_id(self, author_name: str, series_id: str) -> List[Any]:
        series_id = str(series_id)
        texts = self.get_books_by_author(author_name)
        text = [book for book in texts if series_id == book.id_in_series
        .values[0]]
        return text[0]

    def get_book_by_name(self, author_name: str, book_name: str) -> List[Any]:
        texts = self.get_books_by_author(author_name)
        text = [book for book in texts if book_name.lower() in book.header_0.str.lower().values[0]] \
            if 'book_name' not in texts[0].columns \
            else [book for book in texts if book_name.lower() in book.book_name.values[0].lower()]
        return text[0]

    def get_sentences(self, chapter) -> List[Any]:
        sentences = self.__get_sentences_ids_chapter(chapter)[0]
        return sentences

    def get_sentences_with_norm_form(self, chapter) -> Tuple[List[Any], List[Any]]:
        sentences, sentences_ids, chptr = self.__get_sentences_ids_chapter(chapter)
        morf_sentences = chptr.pymorphy
        norm_form_sentences = [morf_sentences.loc[sentence]['normal_form'] for sentence in sentences_ids]
        return sentences, norm_form_sentences

    def get_sentences_with_stem_form(self, chapter) -> Tuple[List[Any], List[Any]]:
        sentences, sentences_ids, chptr = self.__get_sentences_ids_chapter(chapter)
        morf_sentences = chptr.snowball
        norm_form_sentences = [morf_sentences.loc[sentence]['stem'] for sentence in sentences_ids]
        return sentences, norm_form_sentences
