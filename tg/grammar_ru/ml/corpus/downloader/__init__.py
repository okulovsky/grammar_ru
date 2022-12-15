from typing import *
from yo_fluq_ds import *
import requests
import time
import browser_cookie3
from pathlib import Path

def download(
        url_pattern: str,
        folder: Path,
        values: List[str],
        pause_time: float = 1,
        dont_redownload: bool = True,
        with_progress_bar: bool = True,
        continue_if_not_found: bool = True,
        extension = '.html',
        cookies_for_domain = None
):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    if cookies_for_domain is not None:
        cookies = browser_cookie3.firefox(domain_name=cookies_for_domain)
    else:
        cookies = None

    vs = Query.en(values)
    if with_progress_bar:
        vs = vs.feed(fluq.with_progress_bar())

    for value in vs:
        fvalue = value.replace('/','___')
        path = folder/(fvalue+extension)
        os.makedirs(path.parent, exist_ok=True)
        if path.is_file():
            if dont_redownload:
                continue
        url = url_pattern.format(value)
        time.sleep(pause_time)
        response = requests.get(url, headers=headers, cookies=cookies)
        if response.status_code!=200:
            msg = f'Error when accessing {url}: status {response.status_code}'
            if not continue_if_not_found:
                raise ValueError(msg)
            else:
                print(msg)
                continue
        FileIO.write_text(response.text, path)

