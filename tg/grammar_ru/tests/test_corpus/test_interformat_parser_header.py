from typing import *
from tg.grammar_ru.ml.corpus.formats.interformat_parser import HeaderParser, HeaderParseResponse
from unittest import TestCase



class Sz:
    def __init__(self, tc: 'HeaderTestCase'):
        self.tc = tc
        self.parser = HeaderParser()

    def test(self, input, resp: Optional[HeaderParseResponse] = None, expected_stack = None) -> 'Sz':
        result = self.parser.observe(input)
        if resp is not None:
            self.tc.assertEqual(resp, result)
        if expected_stack is not None:
            self.tc.assertListEqual(expected_stack, [z[1] for z in self.parser.stack])
        return self

    def test1(self, input, expected_stack = None) -> 'Sz':
        return self.test(input, None, expected_stack)

    def test2(self, input, **tags):
        result = self.parser.observe(input)
        if len(tags)>0:
            c = self.parser.get_header_tags()
            del c['headers']
            self.tc.assertDictEqual(c, tags)
        return self


class HeaderTestCase(TestCase):
    def test_blocks(self):
        (Sz(self)
         .test('x', HeaderParseResponse.NewTextBlock, [])
         .test('x', HeaderParseResponse.ContinueTextBlock, [])
         .test('# a', HeaderParseResponse.Ignore, ['a'])
         .test('   ', HeaderParseResponse.Ignore, ['a'])
         .test('# b', HeaderParseResponse.Ignore, ['a','b'])
         .test('xx', HeaderParseResponse.NewTextBlock, ['a','b'])
         .test('yy', HeaderParseResponse.ContinueTextBlock, ['a','b'])
        )

    def test_cont_header(self):
        (Sz(self)
         .test('#a', HeaderParseResponse.Ignore, ['a'])
         .test('#b', HeaderParseResponse.Ignore, ['a','b'])
         .test(' ', HeaderParseResponse.Ignore,['a','b'])
         .test('#c', HeaderParseResponse.Ignore, ['a', 'b', 'c'])

         )

    def test_next_header(self):
        (Sz(self)
         .test('# a', HeaderParseResponse.Ignore, ['a'])
         .test(' xx ', HeaderParseResponse.NewTextBlock, ['a'])
         .test('# b', HeaderParseResponse.Ignore, ['b'])
        )


    def test_multi_header(self):
        (Sz(self)
         .test1('#a', ['a'])
         .test1('## b', ['a', 'b'])
         .test1('x')
         .test1('## c', ['a', 'c'])
         .test1('z')
         .test1('#n', ['n'])
        )

    def test_skipped_levels(self):
        (Sz(self)
         .test1('#a', ['a'])
         .test1('### b', ['a','b'])
         .test1('x')
         .test1('## c', ['a','c'])
         )

    def test_json_header(self):
        (Sz(self)
         .test2('# a', header_0='a')
         .test2('$ {"x": 56} ', header_0='a', tag_x=56)
         .test2('test')
         .test2('test', header_0='a', tag_x=56)
         .test2('## b', header_0 = 'a', header_1='b', tag_x=56)
         .test2('$ {"y":100}', header_0='a', header_1='b', tag_y=100)
        )