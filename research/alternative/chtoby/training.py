from pathlib import Path
import click

from tg.grammar_ru.common import Loc
from research.common import run_training


@click.command()
@click.argument('bundle_name')
def cli(bundle_name: str) -> None:
    bundle_path = Loc.bundles_path/f'chtoby/{bundle_name}'

    run_training.run_local(Path(__file__)/bundle_path)


if __name__ == '__main__':
    cli()

