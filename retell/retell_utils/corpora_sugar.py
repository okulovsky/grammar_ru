import numpy as np
from typing import Any, Tuple
from typing import List


class CorpusSugar:
    def __init__(self, corpus_book, corpus_retell):
        self.corpus_book = corpus_book
        self.corpus_retell = corpus_retell
        self.mapping_retell_to_text = corpus_retell.read_mapping_data()

    # def get_data_by_key(self, key: str, value: tp.Any, mutated_coprus: pd.DataFrame = None):
    #     df = self.corpus_df if mutated_coprus is None else mutated_coprus
    #     return df[df[key] == value]

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

    def get_sentences(self, chapter, corpus_type="book") -> List[Any]:
        sentences = self.__get_sentences_ids_chapter(chapter, corpus_type)[0]
        return sentences

    def get_sentences_with_norm_form(self, chapter, corpus_type="book") -> Tuple[List[Any], List[Any]]:
        sentences, sentences_ids, chptr = self.__get_sentences_ids_chapter(chapter, corpus_type)
        morf_sentences = chptr.pymorphy
        norm_form_sentences = [morf_sentences.loc[sentence]['normal_form'] for sentence in sentences_ids]
        return sentences, norm_form_sentences

    def get_true_retell(self, author_name: str, retell_type, retell_detail):
        if retell_type not in ['retell', 'summary']:
            raise ValueError('Retell type must be retell of summary')
        books, retell_books, summar_books = self.get_books_retell_info_by_author(author_name)
        to_retell = retell_books if retell_type == 'retell' else summar_books
        true_retell = []
        for book in to_retell[:1]:
            for chapter in book.index:
                sentences = self.get_sentences(chapter, "retell")
                true_retell.append("\n".join(" ".join(sentence) for sentence in sentences[:retell_detail]))
        return true_retell
