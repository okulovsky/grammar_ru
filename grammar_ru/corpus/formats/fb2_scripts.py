from yo_fluq_ds import *
from ...common import FileIO
from .fb2_parser import *


class Converter:
    def __init__(self):
        self.result = []
        self.buffer = ''
        self.state = 'waiting'

    def flush(self):
        if self.buffer != '':
            self.result.append(self.buffer)
        self.buffer = ''

    def observe(self, rec: Record):
        if rec.path == 'FictionBook/body':
            if self.state == 'waiting':
                self.state = 'processing'
            else:
                self.state = 'stopped'
        if self.state=='stopped':
            return
        reg = Fb2Parser.detect_regex(rec.path)
        if reg == 'newline':
            self.flush()
            self.buffer = rec.text or ''
        elif reg == 'sameline':
            self.buffer += rec.text or ''
        elif reg == 'header':
            self.flush()
            header_count = 0
            for header_count in range(4):
                if '/section' * header_count not in rec.path:
                    break
            if '/subtitle' in rec.path:
                header_count += 1
            self.buffer = ('#' * (header_count + 1)) + ' ' + (rec.text or '')
        elif reg == 'ignore':
            pass
        else:
            if not rec.is_empty:
                raise ValueError(f'{rec.path} is not recognized by regexes')
        # print(f"{rec.path}   {reg}/{rec.text}/{self.buffer}")

class Fb2Scripts:
    @staticmethod
    def get_parse_dataframe(path):
        return (Fb2Parser
                .linear_load_folder(path)
                .select(lambda z: (z.path, Fb2Parser.detect_regex(z.path), z.is_empty))
                .feed(fluq.count_by(lambda z: z))
                .select(lambda z: z.key + (z.value,))
                .to_dataframe(columns=['path', 'command', 'is_empty', 'count'])
                )

    @staticmethod
    def check_parse_dataframe(df):
        df = df.loc[df.command.isnull() & ~df.is_empty]
        if df.shape[0] == 0:
            return None
        return df

    @staticmethod
    def get_body_list(path):
        bodies = []
        for rec in Fb2Parser.linear_load_folder(path):
            if rec.path == 'FictionBook/body':
                bodies.append(dict(path=rec.filename.name, ord=len(bodies), title='', length=0))
            if rec.path.startswith('FictionBook/body/title'):
                tit = '' if rec.text is None else rec.text
                tit = tit.replace('\n', '')
                bodies[-1]['title'] += tit
            if rec.path.startswith('FictionBook/body'):
                bodies[-1]['length'] += 0 if rec.text is None else len(rec.text)
        df = pd.DataFrame(bodies)
        df = df.feed(fluq.add_ordering_column('path', 'ord', 'body_index'))
        df = df.feed(fluq.add_ordering_column('path', ('length', False), 'size_index'))
        df = df.drop('ord', axis=1)
        df = df.sort_index()
        return df

    @staticmethod
    def check_body_list(df):
        many_bodies = df.loc[df.body_index > 1]
        if many_bodies.shape[0] != 0:
            return df.loc[df.path.isin(many_bodies.path)]

        wrong_order = df.loc[df.body_index!=df.size_index]
        if wrong_order.shape[0] != 0:
            return df.loc[df.path.isin(wrong_order.path)]

        return None

    @staticmethod
    def convert_file(file_path: Path, source_folder: Path, dst_folder: Path):
        cnv = Converter()
        Fb2Parser.linear_load(file_path).foreach(cnv.observe)
        cnv.flush()

        end = str(file_path.relative_to(source_folder))
        end = end.replace('.zip','.md')

        dst_path = dst_folder/end
        if not dst_path.parent.is_dir():
            os.makedirs(dst_path.parent)
        FileIO.write_text('\n'.join(cnv.result), dst_path)

    @staticmethod
    def convert_all(source_folder: Path, dst_folder: Path):
        Query.folder(source_folder,'**/*.zip').foreach(lambda z: Fb2Scripts.convert_file(z, source_folder, dst_folder))


