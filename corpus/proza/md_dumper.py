import json
import re
from typing import List

from mdutils.mdutils import MdUtils
from pathlib import Path
from corpus.proza.entities.book import Book


class MdDumper:
    def __init__(self, storage_dir: str):
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.storage_dir = storage_dir

    def dump(self, books: List[Book], col_name, col_url, author_url, total_length, author_info):
        f_name = self.get_file_name(author_url, col_name, self.storage_dir)
        mdFile = MdUtils(file_name=f_name)
        mdFile.new_header(level=1, title=col_name)
        mdFile.write("$ " + json.dumps(dict({'author_url': author_url,
                                             'length': total_length,
                                             'col_name': col_name,
                                             'collection_url': col_url,
                                             }, **author_info)))
        b = False

        for chapter in books:
            mdFile.new_header(level=2, title=chapter.name)
            mdFile.write("$ " + json.dumps({'book_name': chapter.name,
                                            'url': chapter.rel_url,
                                            'review_count': chapter.review_cnt,
                                            'publication_date': chapter.publication_date.strftime('%d.%m.%Y')}))
            p = re.compile('\s\*\s?\*\s?\*\s')
            for i, part in enumerate(p.split(chapter.content)):
                mdFile.new_header(level=3, title=f'star part {i}')
                mdFile.write("$ " + json.dumps({'part_number': i, 'part_length': len(part)}))
                mdFile.write(part)
                # if i > 1:
                #     print(f'partial {f_name}')
                #     b = True
        mdFile.create_md_file()
        # if b: exit()
        print('dumped')
        # print(f'dumped to md {f_name} {len(books)} chapters')

    @staticmethod
    def get_file_name(author_rel_url, col_name, storage_dir):
        forbidden = """<>:"/\|?*"""
        return storage_dir + "/" + "".join(
            [x for x in col_name + author_rel_url.replace('/avtor', ' ') if x not in forbidden])
