from pathlib import Path

import pandas as pd
from tg.common._common.data_bundle import DataBundle

from tg.grammar_ru.components.yandex_storage.s3_yandex_helpers import S3YandexHandler


def set_mask(bundle_all_decl_location: Path, masked_bundle_location: Path, masks: pd.DataFrame):
    db = DataBundle.load(bundle_all_decl_location)
    db['index'] = pd.merge(db.index, masks, left_on='declension_type',
                           right_index=True).sort_index()
    db = db.copy()
    db.save(masked_bundle_location)


def upload_bundle(bundle_location, dataset_name, bucket, project_name):
    s3path = f'datasphere/{project_name}/datasets/{dataset_name}'
    # try:
    #     S3YandexHandler.create_bucket(bucket)
    # except:
    #     pass
    S3YandexHandler.upload_folder(bucket, s3path, bundle_location)

def _print_thrown(thrown, file):
    with open(file, "a") as myfile:
        for w in thrown:
            myfile.write(f'{w}\n')