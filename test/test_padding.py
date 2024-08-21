import unittest
import numpy as np
import scipy.sparse as ssp

from optimizers.padding import sparse_padding


class TestPadding(unittest.TestCase):
    def test_padding_small(self):
        k = 2
        a = ssp.coo_array([[1, 2], [3, 4]]).tocoo()
        expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        padded_a = sparse_padding(a=a, k=k).toarray()

        self.assertTrue(np.allclose(expected, padded_a))


if __name__ == '__main__':
    unittest.main()
