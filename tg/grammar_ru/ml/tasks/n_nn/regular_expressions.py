import re


endings = r'ыми|ее|их|ых|ая|ого|ей|ом|ему|ие|ые|ую|им|ем|ому|ой|ое|его|ый|ими|ым|ий'
single_n_regex = re.compile(rf'[^н]н(?:{endings})$')
double_n_regex = re.compile(rf'нн(?:{endings})$')
