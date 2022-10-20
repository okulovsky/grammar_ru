import itertools
import os
import pathlib
import re
from datetime import datetime
from typing import Set, List

from tqdm import tqdm

from corpus.proza.CONST import TEXTS_URL, TOPIC, BASE_URL, MONTHS, DAYS, YEARS
from corpus.proza.entities.book import Book
from corpus.proza.html_cacher import HtmlCacher
from corpus.proza.http_client import HttpClient
from corpus.proza.md_dumper import MdDumper

novels_dumped = 0
MDSTORAGE = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/data-cache/md"
HTMLSTORAGE = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/data-cache/html"
html_cacher = HtmlCacher(HTMLSTORAGE)
http_client = HttpClient(html_cacher)
mddumper = MdDumper(MDSTORAGE)
CollectionName = str
url = str

REQUIRED_SIZE = 700_000
# REQUIRED_SIZE = 1_000
REQUIRED_CHAPTER_SIZE = 10_000
# REQUIRED_CHAPTER_SIZE = 10
REQUIRED_PUB_CNT = 10


def get_2_sign(x): return str(x).rjust(2, '0')


def get_authors_books_by_day(topic, year: int, month: int, day: int) -> Set[str]:
    month_2_sign = get_2_sign(month)
    day_2_sign = get_2_sign(day)

    query_params = f"year={year}&month={month_2_sign}&day={day_2_sign}"
    full_date = f"{year}/{month_2_sign}/{day_2_sign}"
    url = TEXTS_URL + f"/list.html?topic={topic}&{query_params}"
    authors_urls = []
    html, soup = http_client.get_html(url)
    for a in soup.find_all('a'):
        href = a.get('href')
        if full_date in href:
            publication_name = a.text
            author_rel_url = a.next_sibling.next_sibling.find("a").get("href")
            authors_urls.append(author_rel_url)
    return set(authors_urls)


p = re.compile('\(\d+\)$')
re_pub = re.compile('.*Произведений.*')
re_rev = re.compile('.*Получено рецензий.*')
re_rev_sent = re.compile('.*Написано рецензий.*')
re_readers = re.compile('.*Читателей.*')


def try_get(a, reg):
    try:
        return int(a.find(text=reg).parent.next_sibling.next_sibling.text)
    except AttributeError:
        return None


def get_author_info(soup, url):
    res = {k: None for k in ["auth_publication_count", "auth_reviews_received_count",
                             "auth_reviews_sent_count", "auth_readers_count"]}
    info_tag = soup.find(text="Читателей")
    if info_tag:
        try:
            a = info_tag.parent.parent
            # res["publication_count"] = int(a.find(text=re_pub).next_sibling.text)
            try:
                res["auth_publication_count"] = int(a.find(text=re_pub).next_sibling.text)
            except AttributeError:
                pass
            res["auth_reviews_received_count"] = try_get(a, re_rev)
            res["auth_reviews_sent_count"] = try_get(a, re_rev_sent)
            res["auth_readers_count"] = try_get(a, re_readers)
        except AttributeError:
            raise Exception(url)
    else:
        pass  # Closed author's page
        # raise Exception("No author info?", url)
    return res


def get_collections_urls(author_rel_url, author_info_by_url):
    url = BASE_URL + f"/{author_rel_url}"
    html, soup = http_client.get_html(url)
    if author_rel_url not in author_info_by_url:
        author_info_by_url[author_rel_url] = get_author_info(soup, url)
    collections_tags = soup.find_all("div", {"id": "bookheader"})
    collections = []
    for tag in collections_tags:
        child = tag.findChildren('a')[0]
        href = child.get("href")
        if not href:
            continue
        collection_name_w_cap = tag.text
        x = p.findall(collection_name_w_cap)
        if x:
            cnt_substr = x[0]
            collection_publications_cnt = int(cnt_substr[1:-1])
            collection_name = collection_name_w_cap[:-len(cnt_substr) - 1]
            collections.append((collection_name, collection_publications_cnt, href))
    return collections


def get_books_from_collection(collection_url, collection_name: CollectionName) -> List[Book]:
    books = []
    html, soup = http_client.get_html(BASE_URL + f"/{collection_url}")
    for y in soup.find_all("div", {"id": "bookheader"}):
        found_collection = y.find(text=collection_name)
        if found_collection:
            book_list = y.next_sibling.next_sibling
            children = book_list.findChildren()
            for book_tag in children:
                a = book_tag.find('a')
                if a:
                    small_text = book_tag.find('small').text
                    publication_date_time_str = small_text.split(',')[1].lstrip().rstrip()
                    category = small_text.split(',')[0].lstrip().rstrip()[2:]
                    if category != 'фэнтези': return []
                    try:
                        dt = datetime.strptime(publication_date_time_str, '%d.%m.%Y %H:%M')
                    except ValueError:
                        dt = datetime.strptime(publication_date_time_str, '%d.%m.%Y')
                    books.append(Book(a.text, a.get("href"), dt))
    return sorted(books, key=lambda b: b.publication_date)


def get_book_content(book: Book):
    review_cnt = 0
    html, soup = http_client.get_html(BASE_URL + book.rel_url)
    notes_tag = soup.find(text="На это произведение написаны ")
    # notes_tag = soup.find("div", {"class": "notesline", "text": "На это произведение написаны "})
    if notes_tag:
        review_cnt = int(notes_tag.parent.find('b').text.split()[0])
    try:
        t = soup.find("div", {"class": "text"}).text.replace('\xa0', ' ')
        return t, review_cnt
    except AttributeError:
        return "", review_cnt


def dump_if_large(books, col_url, col_name, author_url, author_info_by_url) -> bool:
    # книги из одной коллекции
    books.sort(key=lambda b: b.publication_date)
    # print(books)
    f_name = mddumper.get_file_name(author_url, col_name, MDSTORAGE) + ".md"
    if os.path.isfile(f_name):
        print(f"skip {f_name}")
        return True
    for b in books:
        b.content, b.review_cnt = get_book_content(b)
        if len(b.content) < REQUIRED_CHAPTER_SIZE: return False
    total_length = sum(len(b.content) for b in books)
    if total_length > REQUIRED_SIZE:
        mddumper.dump(books, col_name, col_url, author_url, total_length, author_info_by_url[author_url])
        print(f'dumped {f_name}')
        return True
    # print(f'candidate too small {BASE_URL + col_url}. total_size = {total_length}')
    return False


seen_authors = set()
author_info_by_url = {}
for year, month, day in tqdm(list(itertools.product(YEARS, MONTHS, DAYS)), ncols=80, desc="DAYS"):
    print(f'DUMPED {novels_dumped}')
    authors = get_authors_books_by_day(TOPIC.FANTASY, year, month, day)
    # pbar = tqdm(authors - seen_authors, ncols=80, leave=False, desc=f"day = {day}, month = {month} ")
    pbar = authors - seen_authors
    for author_url in pbar:
        seen_authors.add(author_url)
        collections = get_collections_urls(author_url, author_info_by_url)
        for name, publications_cnt, url in collections:
            books = get_books_from_collection(url, name)
            if not books:
                continue
            novels_dumped += dump_if_large(books, url, name, author_url, author_info_by_url)
            if novels_dumped > 150:
                print('breaked')
                exit()
