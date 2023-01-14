from tg.common.tools import ImportFixer
from yo_fluq_ds import *
from tg.common import Loc

def filter(fname):
    text = FileIO.read_text(fname)
    if 'from unittest' in text:
        return False
    if 'if __name__' in text:
        return False
    if 'import unittest' in text:
        return False
    return True


if __name__ == '__main__':
    fixer = ImportFixer(Loc.tg_path, 'tg')

    all_files = Query.folder(Loc.tg_path, '**/*.py').to_list()
    lib_files = Query.en(all_files).where(filter).to_list()
    imps = Query.en(lib_files).select_many(ImportFixer.parse_imports_from_file).select(fixer.fix_absolute_import).where(lambda z: z is not None).to_list()
    files = Query.en(imps).select(lambda z: z.file).distinct().order_by(lambda z: z).to_list()
    Query.en(files).foreach(print)
    (Query
     .en(imps)
     #.where(lambda z: 'grammar_ru/__init__' in str(z.file))
     .foreach(fixer.apply)
     )