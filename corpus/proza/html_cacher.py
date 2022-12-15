import base64
import pathlib
from hashlib import sha256
from pathlib import Path


class HtmlCacher:
    def __init__(self, cache_dir):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

    def _encode_url(self, url: str):
        return sha256(url.encode()).hexdigest() + ".html"

    def save_html_by_url(self, url, html: str):
        file_name = self._encode_url(url)
        full_path = self.cache_dir / file_name
        with open(full_path, 'w') as f:
            f.write(html)

    def get_html_by_url(self, url):
        file_name = self._encode_url(url)
        full_path = self.cache_dir / file_name
        if not Path(full_path).is_file():
            return None
        with open(full_path, 'r') as f:
            return f.read(-1)
