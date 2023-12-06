from tg.grammar_ru.algorithms import alternative
from tg.common import DataBundle


def main():
    text = input()
    alternative_algorithm = alternative.AlternativeAlgorithm()
    result = alternative_algorithm.run_on_string(text)
    print(result)


if __name__ == '__main__':
    main()
