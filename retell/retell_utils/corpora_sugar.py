import numpy as np
from typing import Any, Tuple
from typing import List


class CorpusSugar:
    def __init__(self, corpus_book, corpus_retell, author_name=''):
        self.corpus_book = corpus_book
        self.corpus_retell = corpus_retell
        self.mapping_retell_to_text = corpus_retell.read_mapping_data()
        self.author_name = author_name
        self.books = None
        self.retell_books = None
        self.summar_books = None

    def get_books_retell_info_by_author(self, author_name: str) -> (List[Any], List[Any], List[Any]):
        """
        Take author name and return tuple of: books_info, retell_books_info, summary_books_info
        """
        retell_author_df = self.corpus_retell.get_toc()
        all_retell_author_df = retell_author_df[retell_author_df['author'] == author_name]
        retell_author_df = all_retell_author_df[all_retell_author_df['text_type'] == 'retell']
        summar_martin_df = all_retell_author_df[all_retell_author_df['text_type'] == 'summary']
        retell_books = [retell_author_df[retell_author_df['book_name'] == name_of_book] for name_of_book in
                        retell_author_df['book_name'].unique()]
        summar_books = [summar_martin_df[summar_martin_df['book_name'] == name_of_book] for name_of_book in
                        summar_martin_df['book_name'].unique()]
        author_df = self.corpus_book.get_toc()
        author_df = author_df[author_df['author'] == author_name]
        books = []
        for ret_book in retell_books:
            text_book_indx = self.mapping_retell_to_text.loc[ret_book.index].values.reshape(-1)
            text_book_chapters = author_df.loc[text_book_indx]
            books.append(text_book_chapters)
        self.books = books
        self.retell_books = retell_books
        self.summar_books = summar_books
        return books, retell_books, summar_books

    def __get_sentences_ids_chapter(self, chapter, corpus_type="book") -> Tuple[List[Any], List[Any], Any]:
        corpus = self.corpus_book if corpus_type == "book" else self.corpus_retell
        chptr = corpus.get_bundles([chapter]).single()
        chapter_id = chptr.src
        sentences_id = np.array(chapter_id['sentence_id'].unique())
        sentences_ids = [chapter_id['word_id'][chapter_id['sentence_id'] == sentence_id]
                         for sentence_id in sentences_id]
        sentences = [chapter_id['word'].values[chapter_id['sentence_id'] == sentence_id]
                     for sentence_id in sentences_id]
        return sentences, sentences_ids, chptr

    def __check_call_get_book_info(self, author_name: str) -> None:
        if self.books is None and self.retell_books is None:
            if author_name is None:
                raise ValueError('''You have not extracted book_info before. 
                        Specify the author's name, or call the method get_books_retell_info_by_author.''')
            else:
                self.get_books_retell_info_by_author(author_name)

    def get_book_by_name(self, book_name: str, corpus_type="book", author_name=None) -> List[Any]:
        self.__check_call_get_book_info(author_name)
        texts = self.books if corpus_type == "book" else (
            self.retell_books if corpus_type == 'retell' else self.summar_books)
        text = [book for book in texts if book_name.lower() in book.header_0.str.lower().values[0]] \
            if 'book_name' not in texts[0].columns \
            else [book for book in texts if book_name in book.book_name.values]
        return text[0]

    def get_sentences(self, chapter, corpus_type="book") -> List[Any]:
        sentences = self.__get_sentences_ids_chapter(chapter, corpus_type)[0]
        return sentences

    def get_sentences_with_norm_form(self, chapter, corpus_type="book") -> Tuple[List[Any], List[Any]]:
        sentences, sentences_ids, chptr = self.__get_sentences_ids_chapter(chapter, corpus_type)
        morf_sentences = chptr.pymorphy
        norm_form_sentences = [morf_sentences.loc[sentence]['normal_form'] for sentence in sentences_ids]
        return sentences, norm_form_sentences

    def get_sentences_with_stem_form(self, chapter, corpus_type="book") -> Tuple[List[Any], List[Any]]:
        sentences, sentences_ids, chptr = self.__get_sentences_ids_chapter(chapter, corpus_type)
        morf_sentences = chptr.snowball
        norm_form_sentences = [morf_sentences.loc[sentence]['stem'] for sentence in sentences_ids]
        return sentences, norm_form_sentences

    def get_true_retell(self, author_name: str, retell_type, retell_detail, book_name=None):
        if retell_type not in ['retell', 'summary']:
            raise ValueError('Retell type must be retell of summary')
        self.__check_call_get_book_info(author_name)
        book = self.retell_books if retell_type == 'retell' else self.summar_books
        book = self.get_book_by_name(book_name, retell_type, author_name) \
            if book_name is not None \
            else book[0]
        true_retell = []
        for chapter in book.index:
            sentences = self.get_sentences(chapter, "retell")
            true_retell.append("\n".join(" ".join(sentence) for sentence in sentences[:retell_detail]))
        return true_retell
