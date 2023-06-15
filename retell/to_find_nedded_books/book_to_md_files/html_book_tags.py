def Garry_Potter_retell_tag(tag):
    is_a_tag = tag.name == 'a'
    ban_words = ["Книга", "Игра", "Саундтрек", "Фильм", "Саундтрек (часть 1)", "Саундтрек (часть 2)"]
    has_no_class_atrr = not tag.has_attr('class')
    class_attr_is_mw, isnt_template = True, True
    havent_ban_words = all(ban_word not in tag.contents for ban_word in ban_words)
    if tag.has_attr('class'):
        class_attr_is_mw = "mw-redirect" in tag.attrs['class']
    if tag.has_attr('title'):
        isnt_template = "шаблон:главы" not in tag.attrs['title'].lower()
    return is_a_tag and (has_no_class_atrr or class_attr_is_mw) and havent_ban_words and isnt_template


def witcher_retell_tag(tag):
    is_p_tag = tag.name == 'p'
    only_one_content = len(tag.contents) == 1
    return is_p_tag and only_one_content


def Checkov_retell_tag(tag):
    is_p_tag = tag.name == 'p'
    parent_tag = tag.parent
    isnt_end = 'За основу пересказа' not in tag.text
    parent_div_is_not_poem = False
    if parent_tag.has_attr('class'):
        parent_div_is_not_poem = parent_tag.attrs['class'][0] != 'poem'
    return is_p_tag and isnt_end and parent_div_is_not_poem


def Chekov_book_tag(tag):
    is_a_tag = tag.name == 'a'
    author_name_in_a = False
    if tag.has_attr('title'):
        author_name_in_a = '(Чехов)' in tag.attrs['title']
    return is_a_tag and author_name_in_a
