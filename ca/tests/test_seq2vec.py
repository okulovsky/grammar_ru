import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase
from tg.ca.Seq2VecMatching import Seq2VecMatcher


class Seq2VecTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pad_by_ones = lambda arr: np.pad(arr, ((1, 0), (1, 0)), 'constant')
        self.converter = Seq2VecMatcher()
        up, diag = self.converter._current_group_flag, self.converter._next_group_flag
        self.correct_actions_5x3 = pad_by_ones(np.array([
            [0, 0, 0],
            [up, diag, 0],
            [up, diag, diag],
            [up, up, diag],
            [up, up, up]
        ], dtype=np.float32))
        self.correct_actions_8x5 = pad_by_ones(np.array([
            [0, 0, 0, 0, 0],
            [up, diag, 0, 0, 0],
            [up, up, diag, 0, 0],
            [up, up, diag, diag, 0],
            [up, up, up, diag, diag],
            [up, up, up, diag, diag],
            [up, up, up, diag, diag],
            [up, up, up, up, diag],
        ], dtype=np.float32))

    def test_action_matrix(self):
        simple_cos_sim_mock_5x3 = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1]
        ])
        complicated_cos_sim_mock_8x5 = np.array([
            [1, 0, 0, 0, 0],
            [.5, .9, 0, 0, 0],
            [.1, .8, .4, 0, 0],
            [.3, .1, 1., .4, 0],
            [.3, .4, .7, .4, .3],
            [.3, .4, .9, .2, .3],
            [.3, .4, .5, .8, .3],
            [.3, .4, .5, .6, .6],
        ])
        action_matrix_5x3, action_matrix_8x5 = [self.converter._get_action_matrix(cos_sim) for cos_sim in
                                                [simple_cos_sim_mock_5x3, complicated_cos_sim_mock_8x5]]
        assert_array_equal(action_matrix_5x3, self.correct_actions_5x3)
        assert_array_equal(action_matrix_8x5, self.correct_actions_8x5)

    def test_get_monotone_matching(self):
        correct_monotone_matching_5x3 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
        correct_monotone_matching_8x5 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 4}
        monotone_matching_5x3, monotone_matching_8x5 = [self.converter._get_monotone_matching(action) for action in
                                                        [self.correct_actions_5x3, self.correct_actions_8x5]]
        assert correct_monotone_matching_5x3 == monotone_matching_5x3
        assert correct_monotone_matching_8x5 == monotone_matching_8x5
