import pandas as pd
import typing as tp

from typing import List


class CorpusSugar:
    def __init__(self, corpus_book, corpus_retell):
        self.corpus_book = corpus_book
        self.corpus_retell = corpus_retell
        self.mapping_retell_to_text = corpus_retell.read_mapping_data()

    # def get_data_by_key(self, key: str, value: tp.Any, mutated_coprus: pd.DataFrame = None):
    #     df = self.corpus_df if mutated_coprus is None else mutated_coprus
    #     return df[df[key] == value]

    def get_books_retell_info_by_author(self, author_name: str) -> (List[tp.Any], List[tp.Any], List[tp.Any]):
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
