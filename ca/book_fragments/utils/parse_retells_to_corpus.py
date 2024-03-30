import json
from pathlib import Path
from grammar_ru.common import Separator
from grammar_ru.corpus.corpus_writer import CorpusWriter


def parse_retells_to_corpus(
    fragments_path: Path,
    retells_path: Path,
    corpus_path: Path,
    separator=Separator
):
    corpus_writer = CorpusWriter(corpus_path)

    retells_data = None
    with open(retells_path, 'r') as retells_file:
        retells_data = json.load(retells_file)

    fragments_data = None
    with open(fragments_path, 'r') as fragments_file:
        fragments_data = json.load(fragments_file)

    cur_file_id = ''
    cur_fragment_retells = []
    for fragment in fragments_data["fragments"]:
        fragment_id = fragment["id"]
        if cur_file_id != fragment["file_id"]:
            if cur_file_id != '':
                corpus_writer.add_fragment(
                    separator.separate_string(
                        '\n'.join(cur_fragment_retells)
                    ),
                    file_name=cur_file_id
                )
                cur_fragment_retells = []
            cur_file_id = fragment["file_id"]

        fragment = retells_data[fragment_id]["choices"][0]["message"]["content"]
        cur_fragment_retells.append(_prettify_fragment(fragment))
    
    if len(cur_fragment_retells) != 0:
        corpus_writer.add_fragment(
            separator.separate_string(
                '\n'.join(cur_fragment_retells)
            ),
            file_name=cur_file_id
        )
        cur_fragment_retells = []
    
    corpus_writer.finalize()
        

def _prettify_fragment(fragment: str):
    fragment_points = fragment.split('\n')

    result_points = []
    for fragment_point in fragment_points:
        fragment_point = fragment_point[3:]
        if len(fragment_point) > 0 and fragment_point[-1] != '.':
            fragment_point += '.'
        result_points.append(fragment_point)

    return ' '.join(result_points)
