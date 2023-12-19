from tg.common import DataBundle, Loc

db = DataBundle.load(r'C:\Users\Nikolai\PycharmProjects\grammar_ru\demos\files\tsa-test.zip')

print(Separator.Viewer().to_text(db.src.loc[db.src.sentence_id==1]))