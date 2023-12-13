import requests, re, os
from bs4 import BeautifulSoup
from pathlib import Path

from tg.projects.retell.md_book_from_url.data_utils.books_dataclasses import BookInfo, ChapterInfo
from filter_html_tags import Garry_Potter_retell_tag, Checkov_retell_tag, Chekov_book_tag, witcher_retell_tag
from typing import List
from itertools import takewhile


def get_parsed_retell_g_p(url: str) -> List[str]:
    collected_data: List[BookInfo] = []
    with requests.get(url) as response:
        soup = BeautifulSoup(response.text)
    books = soup.findAll('table')
    for book in books:
        book_name = book.find('tbody').find('tr').find('th').find('span', {'style': 'font-size:110%'}) \
            .text.replace(' (главы)', '')
        book_info = BookInfo(book_name, [])
        chapters = book.findAll(Garry_Potter_retell_tag)
        for chapter in chapters:
            chapter_name = chapter.attrs['title']  # chapter.contents[0]
            chapter_url = url.replace('Категория:Главы_книг',
                                      chapter_name.replace(' ', '_'))
            with requests.get(chapter_url) as response:
                soup = BeautifulSoup(response.text)
            data = [tag.text for tag in soup.findAll('p')[3:]]
            summary: str = data[0]
            retelling: str = ''.join(data[1:])
            book_info.chapters.append(ChapterInfo(chapter_name, retelling, summary))
        collected_data.append(book_info)
    return collected_data


def get_parsed_retell_witch(url: str) -> List[str]:
    collected_data: List[BookInfo] = []
    with requests.get(url) as response:
        soup = BeautifulSoup(response.text)
    books_names = [tag.contents[2].attrs['title'].replace(' ', '_') for tag in
                   soup.findAll('p', {"class": "read-more"})]
    base_url = "https://wiki.briefly.ru/"
    for book_name in books_names:
        book_info = BookInfo(book_name, [])
        book_url = base_url + book_name
        with requests.get(book_url) as response:
            soup = BeautifulSoup(response.text)
        retail = [tag.text for tag in soup.findAll(witcher_retell_tag)]
        retail_str = ''.join(retail)
        book_info.chapters.append(retail_str)
        collected_data.append(book_info)
    return collected_data


def get_parsed_retell_Checov(url):
    collected_data: List[BookInfo] = []
    with requests.get(url) as response:
        soup = BeautifulSoup(response.text)
    books_names = [tag.contents[0].replace(' ', '_') for tag in soup.findAll(Chekov_book_tag)]
    base_url = "https://wiki.briefly.ru/"
    for book_name in books_names:
        book_info = BookInfo(book_name, [])
        book_url = base_url + book_name
        with requests.get(book_url) as response:
            soup = BeautifulSoup(response.text)
        retail = [tag.text for tag in soup.findAll(Checkov_retell_tag)]
        retail_str = ''.join(retail)
        book_info.chapters.append(retail_str)
        collected_data.append(book_info)
    return collected_data


def get_parsed_retell_game_o_t(url):
    collected_data: List[BookInfo] = []
    with requests.get(url) as response:
        soup = BeautifulSoup(response.text)
    books_names = [tag.contents[0].attrs['title'].replace(' ', '_') for tag in soup.findAll('b')[:5]]
    base_url = "https://7kingdoms.ru/wiki/"
    for book_name in books_names:
        book_info = BookInfo(book_name, [])
        book_url = base_url + book_name
        with requests.get(book_url) as response:
            soup = BeautifulSoup(response.text)

        table = soup.find('table', {"class": "toc plainlinks common-table"})
        chapters = table.find('tbody').findAll('tr')[1].find('td').findAll('a')
        for chapter in chapters:
            chapter_name = chapter.attrs['title'].replace(' ', '_')
            chapter_url = base_url + chapter_name
            with requests.get(chapter_url) as response:
                soup = BeautifulSoup(response.text)
            all_p = soup.findAll('p')
            summary = all_p[0].text.replace('Краткое содержание:', '').strip()
            retell = [all_p[1].text]
            for elem in all_p[1].next_siblings:
                if elem.name == 'h2':
                    break
                if elem.name == 'p':
                    retell.append(elem.text)
            chapter_info = ChapterInfo(chapter_name, ''.join(retell), summary)
            book_info.chapters.append(chapter_info)
        collected_data.append(book_info)
    return collected_data


def __get_sorted_chapters(book_name, chapter_list, first_chapter=None, last_chapter=None):
    chapters = [chapter_name
                for chapter_name in chapter_list
                if chapter_name.startswith(book_name)
                and 'Appendix' not in chapter_name
                and chapter_name != book_name + '.html' and '.html' in chapter_name]

    find_nums = lambda chapter: 0 if first_chapter in chapter else (
        len(chapters) - 1 if last_chapter in chapter
        else int(re.findall(r'\d+', chapter)[0]))

    for i in range(len(chapters)):
        chapters[i] = (find_nums(chapters[i]), chapters[i])
    chapters = list(map(lambda ch: ch[1], sorted(chapters)))
    return chapters


def eng_get_parsed_retell_game_o_t(url):
    summary_path = Path("/home/mixailkys/websites/ONLY CHAPTERS/awoiaf.westeros.org/index.php")
    book_names = ['A_Game_of_Thrones', 'A_Clash_of_Kings', 'A_Storm_of_Swords', 'A_Feast_for_Crows',
                  'A_Dance_with_Dragons']
    htmls = os.listdir(summary_path)
    book_chapters = {book_name: __get_sorted_chapters(book_name, htmls, 'Prologue', 'Epilogue')
                     for book_name in book_names}
    collected_data: List[BookInfo] = []
    for book_name, chapter_names in book_chapters.items():
        book = BookInfo(book_name, [])
        for i, chapter_number in enumerate(chapter_names):
            with open(summary_path / chapter_number, 'r') as chapter:
                soup = BeautifulSoup(chapter)
            info_tag = soup.find('span', {'id': "Synopsis"})
            summary = ''.join([tag.text for tag in takewhile(lambda _tag: _tag.name != 'h2', info_tag.next_elements)
                               if tag.name == 'p'])
            chapter_name = str(soup.find('th', {'colspan': "2"}).contents[0]).replace(' ', '_')
            book.chapters.append(ChapterInfo(chapter_name + '-' + f"Chapter_{i}", summary))
        collected_data.append(book)
    return collected_data
