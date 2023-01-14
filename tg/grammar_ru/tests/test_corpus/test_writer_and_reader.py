from tg.grammar_ru.corpus import CorpusWriter, CorpusReader
from tg.grammar_ru.corpus import InterFormatParser
from unittest import TestCase
from pathlib import Path
import os



class CorpusTestCase(TestCase):
    def test_corpus(self):

        fragments = InterFormatParser(Path('/test'), Path('/test/a/b.md'), ['folder', 'name'], mock = text).parse().to_list()
        corpus_file = Path(__file__).parent/'temp_corpus.zip'
        writer = CorpusWriter(corpus_file, overwrite=True)
        for f in fragments:
            writer.add_fragment(f)
        writer.finalize()


        reader = CorpusReader(corpus_file)
        toc = reader.get_toc()
        self.assertListEqual(['h1 / h1.1']*2, list(toc.header_0))
        self.assertListEqual(['h2','h22'], list(toc.header_1))

        dfs = reader.get_frames(toc.index).to_list()
        print(dfs[0])
        print(dfs[1])
        self.assertListEqual([0,0,0,1,1], list(dfs[0].sentence_id))
        self.assertListEqual([10005,10005], list(dfs[1].sentence_id))

        os.remove(corpus_file)





text  = '''
# h1
# h1.1
## h2
Первый параграф. 

Второй.

## h22

Третий.
'''