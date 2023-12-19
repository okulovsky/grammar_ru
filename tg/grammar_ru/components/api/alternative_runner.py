import torch

from tg.common import DataBundle, Loc
from tg.grammar_ru import Separator
from tg.grammar_ru.algorithms import alternative
from tg.grammar_ru.features import SnowballFeaturizer
from tg.common.ml.batched_training import context as btc
from tg.common.ml import batched_training as bt
from tg.common.ml.batched_training import sandbox as bts



def main():
    alternative_algorithm = alternative.AlternativeAlgorithm()
    text = 'Никуда не годится если нет. Пить ходить гулять водиться. Нечем больше поделиться. воды напится сражаться красоваться метается вылиться красится нравится.'
    db = alternative_algorithm.create_db(text)
    index = alternative_algorithm.create_index(db, text)
    result = alternative_algorithm.run(db, index)
    print(result)




if __name__ == '__main__':
    main()
