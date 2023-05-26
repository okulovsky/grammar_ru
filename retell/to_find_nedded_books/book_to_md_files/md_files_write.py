from pathlib import Path
import re


def write_gp_md(gp):
    for book in gp:
        with open(Path(f'./parsed_data/gp/{book.name.replace("_", " ")}_summary.md'), 'w') as file:
            for chapter in book.chapters:
                if 'В настоящее время на этой странице нет текста.' not in chapter.summary and chapter.summary != '\n':
                    file.write(f"##{chapter.name}\n\n{chapter.summary}\n")
        with open(Path(f'./parsed_data/gp/{book.name.replace("_", " ")}_retell.md'), 'w') as file:
            for chapter in book.chapters:
                if len(chapter.retell) > 0:
                    file.write(f"##{chapter.name}\n\n{chapter.retell}\n")


def write_witcher_md(witcher):
    for book in witcher:
        with open(Path(f'./parsed_data/witcher/{book.name.replace("_", " ")}.md'), 'w') as file:
            file.write(f"##{book.name}\n\n{book.chapters[0]}\n")


def write_Checov_md(Checov):
    for book in Checov:
        with open(Path(f'./parsed_data/Chekhov/{book.name.replace("_", " ")}.md'), 'w') as file:
            file.write(f"##{book.name}\n\n{book.chapters[0]}\n")


def write_game_o_trones_md(game_o_trones):
    for book in game_o_trones:
        with open(Path(f'./parsed_data/game_of_t/{book.name.replace("_", " ")}_summary.md'), 'w') as file:
            for chapter in book.chapters:
                file.write(
                    f"##{chapter.name}\n\n{chapter.summary.replace('Краткое cодержание: ', '').replace('Краткое Содержание: ', '')}\n\n")
        with open(Path(f'./parsed_data/game_of_t/{book.name.replace("_", " ")}_retell.md'), 'w') as file:
            for chapter in book.chapters:
                retell = re.sub("[\(\[].*?[\)\]]", "", chapter.retell)
                file.write(f"##{chapter.name}\n\n{retell}\n")
