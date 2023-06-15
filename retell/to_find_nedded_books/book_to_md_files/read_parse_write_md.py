from parse_html_books import get_parsed_retell_g_p, get_parsed_retell_witch, get_parsed_retell_Checov, \
    get_parsed_retell_game_o_t
from title_urls import gp_url, witcher_url, checkov, game_o_t
from md_files_write import write_gp_md, write_witcher_md, write_Checov_md, write_game_o_trones_md


titles_urls = [gp_url, witcher_url, checkov, game_o_t]
parsers = [get_parsed_retell_g_p, get_parsed_retell_witch, get_parsed_retell_Checov, get_parsed_retell_game_o_t]
md_writers = [write_gp_md, write_witcher_md, write_Checov_md, write_game_o_trones_md]

parsed_data = [parser(url) for url, parser in zip(titles_urls, parsers)]

for data, writer in zip(parsed_data, md_writers):
    writer(data)
