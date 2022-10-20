import asyncio

import aiohttp
import requests as requests
from bs4 import BeautifulSoup
import time
from corpus.proza.html_cacher import HtmlCacher


class HttpClient:
    def __init__(self, html_cacher):
        self.request_period = 0.5  # sec
        self.last_request_time = time.time() - self.request_period * 10
        self.html_cacher: HtmlCacher = html_cacher

    def get_html(self, url: str):
        cached = self.html_cacher.get_html_by_url(url)
        if cached:
            # print("cached", url, len(cached))
            return cached, BeautifulSoup(cached, 'lxml')
        beg = time.time()
        if beg - self.last_request_time < self.request_period:
            time.sleep(self.last_request_time + self.request_period - beg)
        atmpts = 0
        while atmpts < 5:
            try:  # cringe
                html = requests.get(url).text
                break
            except requests.exceptions.ConnectionError:
                pass

        self.html_cacher.save_html_by_url(url, html)
        soup = BeautifulSoup(html, 'lxml')
        end = time.time()
        # print(f"request in " + "{:10.4f}".format(end - beg) + f"s  {url}")
        return html, soup
