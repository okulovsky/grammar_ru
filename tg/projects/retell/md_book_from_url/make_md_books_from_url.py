from tg.projects.retell.md_book_from_url.functional_utils.parse_html_books import get_parsed_retell_g_p, get_parsed_retell_witch, get_parsed_retell_Checov, \
    get_parsed_retell_game_o_t, eng_get_parsed_retell_game_o_t
from tg.projects.retell.md_book_from_url.data_utils.title_urls import gp_url, witcher_url, checkov, game_o_t, \
    eng_game_o_t
from tg.projects.retell.md_book_from_url.functional_utils.write_md_files import write_gp_md, write_witcher_md, write_Checov_md, write_game_o_trones_md, \
    eng_write_game_o_trones_md

titles_urls = [gp_url, witcher_url, checkov, game_o_t, eng_game_o_t]
parsers = [get_parsed_retell_g_p, get_parsed_retell_witch, get_parsed_retell_Checov, get_parsed_retell_game_o_t,
           eng_get_parsed_retell_game_o_t]
md_writers = [write_gp_md, write_witcher_md, write_Checov_md, write_game_o_trones_md, eng_write_game_o_trones_md]

parsed_data = [parser(url) for url, parser in zip(titles_urls, parsers)]

for data, writer in zip(parsed_data, md_writers):
    writer(data)

