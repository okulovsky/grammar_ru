def prettify_fragment(fragment: str):
    fragment_points = fragment.split('\n')

    result_points = []
    for fragment_point in fragment_points:
        fragment_point = fragment_point[3:]
        if len(fragment_point) > 0 and fragment_point[-1] != '.':
            fragment_point += '.'
        result_points.append(fragment_point)

    return ' '.join(result_points)