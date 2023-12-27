from dotenv import load_dotenv

import common
from tg.grammar_ru.common import Loc
from tg.grammar_ru.components.yandex_storage.s3_yandex_helpers import S3YandexHandler


load_dotenv(Loc.root_path / 'environment.env')

bundle_path = Loc.bundles_path/'punct/9kk'

s3path = f'datasphere/{common.project_name}/datasets/{common.datasphere_dataset_name}'
S3YandexHandler.upload_folder(common.datasphere_bucket, s3path, bundle_path)
