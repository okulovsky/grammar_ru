from enum import Enum
from pathlib import Path
from typing import *
from yo_fluq_ds import FileIO, Query, Queryable
from ..corpus_writer import CorpusFragment
from tg.grammar_ru.common import Separator, Symbols
import re
import json


class HeaderParseResponse(Enum):
    ContinueTextBlock = 0
    NewTextBlock = 1
    Ignore = 2

class HeaderParser:
    def __init__(self):
        self.stack = []
        self.last_is_header = True
        self.custom_tags = {}

    def _parse_header(self, s):
        if len(s)>0 and s[0] == '$':
            try:
                self.custom_tags = json.loads(s[1:])
                return -1, None
            except:
                len(s), ''
        for i in range(len(s)):
            if s[i]!='#':
                return i, s[i:]
        return len(s), ''

    def _push_to_stack(self, level, header):
        break_at = -1
        for i in range(len(self.stack)-1, -1, -1):
            if self.last_is_header:
                if self.stack[i][0] <= level:
                    break_at = i
                    break
            if self.stack[i][0] < level:
                break_at = i
                break
        self.stack = self.stack[:break_at+1]
        self.stack.append((level, header))

    def observe(self, s):
        level, header = self._parse_header(s)
        if level==0:
            if self.last_is_header:
                if s.strip()!='':
                    self.last_is_header = False
                    return HeaderParseResponse.NewTextBlock
                else:
                    return HeaderParseResponse.Ignore
            else:
                return HeaderParseResponse.ContinueTextBlock
        else:
            if header is not None:
                self._push_to_stack(level,header.strip())
            self.last_is_header = True
            return HeaderParseResponse.Ignore

    def _get_header_tags_base(self):
        if len(self.stack)==0:
            return {}
        last_level = -1
        current_suffix = 0
        current_header = -1
        result = {}
        for level, header in self.stack:
            if level == last_level:
                current_suffix+=1
            else:
                last_level = level
                current_suffix = 0
                current_header+=1
            if current_header not in result:
                result[current_header] = dict()
            result[current_header][current_suffix] = header.strip()
        return result

    def get_header_tags(self):
        result = (
            Query.dict(self._get_header_tags_base())
            .to_dictionary(
                lambda z: f'header_{z.key}',
                lambda z: ' / '.join(Query.dict(z.value).order_by(lambda x: x.key).select(lambda x: x.value))
            )
        )
        headers = ' / '.join(Query.dict(result).order_by(lambda z: z.key).select(lambda z: z.value))
        result['headers'] = headers
        for key, value in self.custom_tags.items():
            result['tag_'+key] = value
        return result



_apos_regex = re.compile('([{0}])[{1}]([{0}])'.format(re.escape(Symbols.RUSSIAN_LETTERS),re.escape(Symbols.APOSTROPHS)))
_apos_subs = '\\1{0}\\2'.format(chr(8242))

class InterFormatParser:
    def __init__(self, folder: Path, file_path: Path, naming_schema: List[str], mock: Optional[str] = None):
        self.folder = folder
        self.file_path = file_path
        self.naming_schema = naming_schema
        self.mock = mock


    def _get_prefix_tag(self):
        folder = self.folder.absolute()
        file_path = self.file_path.absolute()
        folder_prefix = str(file_path.parent)
        folder_prefix = folder_prefix[len(str(folder))+1:]
        if len(folder_prefix)>0:
            folder_tags = folder_prefix.split('/')
        else:
            folder_tags = []
        name_tags = file_path.name.split('.')[0].split('-')
        tags = folder_tags+name_tags
        if len(self.naming_schema)!=len(tags):
            raise ValueError(f'Naming schema is violated:\n{self.naming_schema}\n{tags}')
        return {k:v for k,v in zip(self.naming_schema,tags)}


    @staticmethod
    def _circumvent_separator_problems(line):
        line = line.replace(chr(173), '')
        line = _apos_regex.sub(_apos_subs,line)
        return line


    def _parse_base(self):
        text = self.mock
        if self.mock is None:
            text = FileIO.read_text(self.file_path)
        current = None
        parser = HeaderParser()
        for line in text.split('\n'):
            line = InterFormatParser._circumvent_separator_problems(line)
            resp = parser.observe(line)
            if resp == HeaderParseResponse.Ignore:
                continue
            if resp == HeaderParseResponse.NewTextBlock:
                if current is not None:
                    yield current
                current = None
            if current is None:
                current = ([], parser.get_header_tags())
            current[0].append(line)
        if current is not None:
            yield current


    def _parse_iter(self):
        file_tags = self._get_prefix_tag()
        rel_path = self.file_path.relative_to(self.folder)
        for index, (buffer, tags) in enumerate(self._parse_base()):
            for k, v in file_tags.items():
                tags[k]=v
            df = Separator.separate_paragraphs(buffer)
            yield CorpusFragment(rel_path, index, df, tags)

    def parse(self) -> Queryable[CorpusFragment]:
        return Query.en(self._parse_iter())













