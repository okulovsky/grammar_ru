from yo_fluq_ds import *
from tg.grammar_ru.common import Loc
import pandas as pd
from io import BytesIO
import re


NROWS = 868646
PARTITION = 1000
NPARTS = 1+NROWS//PARTITION

FILE = Loc.raw_path / 'lenta/lenta-ru-news.csv'
COLUMNS = ['url', 'title', 'text', 'topic', 'tags', 'date']
pd.options.display.max_columns=None
pd.options.display.width=None
ENDING = re.compile('\d\d\d\d/\d\d/\d\d')

def partial_read(partition):
    buffer = []
    for i, text in enumerate(Query.file.text(FILE)):
        if i==0:
             continue
        n = (i-1)//PARTITION
        if n == partition:
            buffer.append(text)
        if n>partition:
            break

    to_remove = []
    for index,text in enumerate(buffer):
        if not text.startswith('https://lenta.ru'):
            to_remove.append(index)
            to_remove.append(index-1)
        end = text[-10:]
        if not ENDING.match(end):
            to_remove.append(index)

    series = pd.Series(buffer)
    series = series.loc[~series.index.isin(to_remove)]
    buffer = list(series)

    partial = '\n'.join(buffer)
    bytes = partial.encode('utf-8')
    io = BytesIO(bytes)
    #with open('test.txt', 'wb') as file:
    #    file.write(bytes)

    return pd.read_csv(io, names = COLUMNS, encoding='utf-8')


def check_file():
    try:
        print(pd.read_csv('test.txt'))
    except pd.errors.ParserError as ex:
        print(type(ex))


def parse_partition(partition):
    result = [f'# PARTITION {partition}']
    df = partial_read(partition)
    #print(df.head())
    for c, text in enumerate(df.text):
        reason = None
        if not isinstance(text, str):
            reason = 'no_string'
        elif text.startswith('#'):
            reason = 'starts #'
        elif text.strip() == '':
            reason = 'empty'
        if reason is not None:
            print(f'{reason} {text}')
        else:
            result.append(text)
    result_str = '\n'.join(result)
    #print(len(result_str))
    path = Loc.processed_path / 'lenta'
    os.makedirs(path, exist_ok=True)
    path = path / f'{partition}.md'
    #print(path)
    FileIO.write_text(result_str, path)
    return df

def parse_to_interformat(start = 0):
    for i in Query.en(range(start, NPARTS)).feed(fluq.with_progress_bar()):
        parse_partition(i)



if __name__ == '__main__':
    parse_to_interformat()
