from yo_fluq_ds import *
from pathlib import Path


if __name__=='__main__':
    deps = Query.file.text('requirements.full.txt').to_list()
    FileIO.write_text('\n'.join(deps), Path(__file__).parent/'tg/common/delivery/packaging/default_requirements.txt')
    print(deps)