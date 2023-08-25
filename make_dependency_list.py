import json
import subprocess
from pprint import pprint

if __name__ == '__main__':
    result = subprocess.check_output(['python', '-m', 'pip', 'freeze'])
    result = result.decode('utf-8').split('\n')
    result = [c for c in result if '@' not in c and c!='' and 'grammar-ru-private' not in c]
    pprint(result)