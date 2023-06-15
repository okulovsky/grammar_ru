import requests
from bs4 import BeautifulSoup
from books_dataclasses import BookInfo, ChapterInfo
from html_book_tags import Garry_Potter_retell_tag, Checkov_retell_tag, Chekov_book_tag, witcher_retell_tag


def get_parsed_retell_g_p(url: str) -> list[str]:
    collected_data: list[BookInfo] = []
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


def get_parsed_retell_witch(url: str) -> list[str]:
    collected_data: list[BookInfo] = []
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
    collected_data: list[BookInfo] = []
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
    collected_data: list[BookInfo] = []
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
