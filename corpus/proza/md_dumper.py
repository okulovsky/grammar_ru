import json
import re
from typing import List

from mdutils.mdutils import MdUtils
from pathlib import Path
from corpus.proza.entities.book import Book


def get_mini_parts(part, bundle_size):
    res = []
    while len(part) > bundle_size:
        m = re.search('\n\s*\n', part[bundle_size:2 * bundle_size])
        if m:
            d = bundle_size + m.start()
        else:
            m2 = re.search('\n', part[bundle_size:2 * bundle_size])
            if m2:
                d = bundle_size + m2.start()
            else:
                return res
        mini_part = part[:d].lstrip().rstrip()
        if mini_part: res.append(mini_part)
        part = part[d + 1:]
    return res + [part]


chapter_delim = re.compile('((\n\s*){3}|\s\*\s?\*\s?\*\s)')


class MdDumper:
    def __init__(self, storage_dir: str, REQUIRED_BUNDLE_PART_SIZE):
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.storage_dir = storage_dir
        self.bundle_size = REQUIRED_BUNDLE_PART_SIZE

    def dump(self, books: List[Book], col_name, col_url, author_url, total_length, author_info):
        f_name = self.get_file_name(author_url, col_name, self.storage_dir)
        mdFile = MdUtils(file_name=f_name)
        mdFile.new_header(level=1, title=col_name)
        mdFile.write("$ " + json.dumps(dict({'author_url': author_url,
                                             'length': total_length,
                                             'col_name': col_name,
                                             'collection_url': col_url,
                                             }, **author_info)))
        for chapter in books:
            mdFile.new_header(level=2, title=chapter.name)
            mdFile.write("$ " + json.dumps({'book_name': chapter.name,
                                            'url': chapter.rel_url,
                                            'review_count': chapter.review_cnt,
                                            'publication_date': chapter.publication_date.strftime('%d.%m.%Y')}))
            splited = chapter_delim.split(chapter.content)
            parts = []
            for p in splited:
                if p is not None:
                    pp = p.strip()
                    if len(pp) > 10: parts.append(pp)
            i = 0
            for part in parts:
                if len(part) > self.bundle_size:
                    for mini_part in get_mini_parts(part, self.bundle_size):
                        self.write_part(mdFile, mini_part, i, f_name)
                        i += 1
                else:
                    self.write_part(mdFile, part, i, f_name)
                    i += 1
        mdFile.create_md_file()

    def write_part(self, mdFile, part, i, f_name):
        # print(f"BUNDLE SIZE {len(part)}")
        if len(part) < 5:
            print(f" tiny bundle {f_name}")
        if len(part) > 2 * self.bundle_size:
            print("HUGE BUNDLE", f_name, len(part), part[:20])
        mdFile.new_header(level=3, title=f'star part {i}')
        mdFile.write("$ " + json.dumps({'part_number': i, 'part_length': len(part)}))
        mdFile.write('\n' + part)

    @staticmethod
    def get_file_name(author_rel_url, col_name, storage_dir):
        forbidden = """<>:"/\|?*"""
        return storage_dir / "".join(
            [x for x in col_name + author_rel_url.replace('/avtor', ' ') if x not in forbidden])
