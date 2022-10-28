from pathlib import Path

from tg.grammar_ru.common import Loc
from research.common import run_training


bundle_name = 'partial'
bundle_path = Loc.bundles_path/f'n_nn/{bundle_name}'

run_training.run_local(Path(__file__)/bundle_path)
