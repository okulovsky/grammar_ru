from typing import *
from yo_fluq_ds import *
import requests
import time
import browser_cookie3
from pathlib import Path
import numpy as np

class DownloadEngine:
    def get(self, url):
        raise NotImplementedError()


class RequestsEngine(DownloadEngine):
    def __init__(self, cookies_for_domain = None):
        self.cookies_for_domain = cookies_for_domain

    def get(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        if self.cookies_for_domain is not None:
            cookies = browser_cookie3.firefox(domain_name=self.cookies_for_domain)
        else:
            cookies = None
        response = requests.get(url, headers=headers, cookies=cookies)
        err = None
        if response.status_code != 200:
            err = response.status_code
        return response.text, err




class Downloader:
    def __init__(self, engine):
        self.engine = RequestsEngine() if engine is None else engine

    def download(
            self,
            url_pattern: str,
            folder: Path,
            values: List[str],
            pause_time: Union[float, Tuple[float,float]] = 1,
            dont_redownload: bool = True,
            with_progress_bar: bool = True,
            continue_if_not_found: bool = True,
            extension = '.html',
            stop_if_filter = None
    ):
        vs = Query.en(values).select(str)
        if with_progress_bar:
            vs = vs.feed(fluq.with_progress_bar())
        first_time = True
        for value in vs:
            fvalue = value.replace('/','___')
            path = folder/(fvalue+extension)
            os.makedirs(path.parent, exist_ok=True)
            if path.is_file():
                if dont_redownload:
                    continue
            url = url_pattern.format(value)
            if not first_time:
                if isinstance(pause_time, float):
                    time.sleep(pause_time)
                else:
                    mx, mn = pause_time
                    t = np.random.rand() * (mx - mn) + mn
                    time.sleep(t)
            first_time = False
            text, err = self.engine.get(url)
            if err is not None:
                msg = f'Error when accessing {url}: status {err}'
                if not continue_if_not_found:
                    raise ValueError(msg)
                else:
                    print(msg)
                    continue
            if stop_if_filter is not None:
                if stop_if_filter(text):
                    break
            FileIO.write_text(text, path)

