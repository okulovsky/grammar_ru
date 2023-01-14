import zipfile
from yo_fluq_ds import Query
import xml
from pathlib import Path
from io import BytesIO
import re

class Record:
    def __init__(self, prefixes, parents, element, tail, filename):
        self.element = element
        if '}' in element.tag:
            self.tag = element.tag.split('}')[1]
        else:
            self.tag = element.tag
        self.text = element.tail if tail else element.text
        self.prefix = prefixes + ('tail',) if tail else prefixes + (self.tag,)
        self.element_path = parents if tail else parents + [element]
        self.filename = filename
        self.is_empty = (self.text is None) or (self.text.strip() == '')
        self.path = '/'.join(self.prefix)


    def __repr__(self):
        return f"{self.path} `{self.text}`"


class WithLuggage:
    def __init__(self, vicinity):
        self.buffer = []
        self.vicinity = vicinity

    def __call__(self, s):
        self.buffer.append(s)
        self.buffer = self.buffer[-self.vicinity:]
        return self.buffer



class Fb2Parser:
    @staticmethod
    def get_root(path: Path):
        with zipfile.ZipFile(path) as file:
            fb2 = Query.en(file.namelist()).where(lambda z: z.endswith('.fb2')).single()
            content = file.read(fb2)
            root = xml.etree.ElementTree.parse(BytesIO(content)).getroot()
            return root

    @staticmethod
    def linearize(path: Path, root, prefix=None, parents=None):
        prefix = prefix or ()
        parents = parents or []
        r = Record(prefix, parents, root, False, path)
        yield r
        for element in list(root):
            for i in Fb2Parser.linearize(path, element, prefix + (r.tag,), parents + [root]):
                yield i
        yield Record(prefix, parents, root, True, path)

    @staticmethod
    def linear_load(path: Path):
        q = Query.en(Fb2Parser.linearize(path, Fb2Parser.get_root(path)))
        return q

    @staticmethod
    def linear_load_folder(path):
        q = (Query
             .folder(path, '**/*.zip')
             .select_many(lambda z: Fb2Parser.linear_load(z))
             )
        return q

    @staticmethod
    def get_vicinity(source, condition,  selector = lambda z: z, vicinity=10, position=5):
        return (source
                .select(WithLuggage(vicinity))
                .where(lambda z: len(z) == vicinity)
                .where(lambda z: condition(z[position]))
                .select(lambda z: [selector(c) for c in z])
                )

    @staticmethod
    def with_vicinity(condition, selector=lambda z: z,  skip=1, vicinity=10, position=5):
        return lambda z: Fb2Parser.get_vicinity(z, condition, selector, vicinity, position)

    MODS = '(/emphasis|/strong|/sup|/tail)+'
    PREF = 'FictionBook/body(/section)*(/cite)?'

    REGEXES = {
        f'{PREF}/p': 'newline',
        f'{PREF}/p{MODS}': 'sameline',

        f'{PREF}/poem/stanza/v': 'newline',
        f'{PREF}/poem/stanza/v{MODS}': 'sameline',

        f'{PREF}/title/p': 'header',
        f'{PREF}/title/p{MODS}': 'sameline',

        f'{PREF}/subtitle': 'header',
        f'{PREF}/subtitle{MODS}': 'sameline',

        '.*(/epigraph).*': 'ignore',
        '.*(/text-author).*': 'ignore',
        '.*/a': 'ignore',
        '.*/a.*': 'ignore',
        '^(?!FictionBook/body).*': 'ignore'
    }

    @staticmethod
    def detect_regex(s):
        for key, value in Fb2Parser.REGEXES.items():
            if re.match(f'^{key}$', s):
                return value
        return None

