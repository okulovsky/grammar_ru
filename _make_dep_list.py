from yo_fluq_ds import *
from pathlib import Path

if __name__ == '__main__':
    text = FileIO.read_text(Path(__file__).parent/'requirements.full.txt')
    print('\n'.join([f"'{c.strip()}'," for c in text.split('\n') if c.strip()!='' and '@' not in c]))