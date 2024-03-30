from tg.grammar_ru.algorithms import alternative


def main():
    alternative_algorithm = alternative.AlternativeAlgorithm()
    text = ('Никуда не годится если нет. Пить ходить гулять водиться. Нечем больше поделиться. воды напится сражаться красоваться метается вылиться красится нравится')
    db = alternative_algorithm.create_db(text)
    result = alternative_algorithm.new_run(db)

    print(result)


if __name__ == '__main__':
    main()
