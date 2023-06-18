import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from libact.utils.multilabel import pairwise_rank_loss, pairwise_f1_score

class MultiLabelCriteriaTestCase(unittest.TestCase):

    def test_criteria(self):
        a = np.array([[1, 0, 1, 0, 0],
                      [1, 1, 0, 1, 0],
                      [1, 0, 0, 1, 0],])
        b = np.array([[0, 1, 1, 0, 0],
                      [0, 1, 0, 0, 1],
                      [1, 0, 1, 1, 0],])
        assert_array_almost_equal(pairwise_rank_loss(a, b), [-2.5, -3.5, -1])
        assert_array_almost_equal(pairwise_f1_score(a, b), [0.5, 0.4, 0.8])
