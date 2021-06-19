from corpus.common import Loc, Corpus
from yo_fluq_ds import *
import shutil
import os
from tg.common.ml import batched_training as bt
from research.amenities.bundles import *


CACHE = Loc.bundles_path / 'repetitions/cache'
FULL =  Loc.bundles_path / 'repetitions/bundle'
TEST = Loc.bundles_path / 'repetitions/test_bundle'

def check_folder(folder, names):
    return Query.en(names).all(lambda z: (folder / z).is_file())

def create_full_bundle():

    sample_folder = Query.folder(CACHE).first()
    names = Query.folder(sample_folder).select(lambda z: z.name).to_list()
    print(names)

    fdf = Query.folder(CACHE).select(lambda z: dict(folder=z, clean=check_folder(z, names))).to_dataframe()
    folders = list(fdf.loc[fdf.clean].folder)
    print(fdf.loc[~fdf.clean])

    shutil.rmtree(FULL, ignore_errors=True)
    os.makedirs(FULL)

    Corpus.get_tocs().to_parquet(FULL / 'toc.parquet')

    for name in names:
        print(name)
        frames = Query.en(folders).feed(fluq.with_progress_bar()).select(lambda z: pd.read_parquet(z / name)).to_list()
        result = pd.concat(frames)
        result.to_parquet(FULL / name)
        del frames
        del result




def create_test_bundle():
    shutil.rmtree(TEST, ignore_errors=True)
    os.makedirs(TEST)

    bundle = bt.DataBundle.load(FULL)
    bundle.index_frame = bundle.index_frame.sample(frac=0.1)
    wids = list(bundle.index_frame.word_id)+list(bundle.index_frame.another_word_id)

    filter_bundle_by_words(bundle, wids)
    bundle.save(TEST)



if __name__ == '__main__':
    create_full_bundle()
    create_test_bundle()


