import unittest
import numpy as np

from partial_gauss import partial_row_reduce


class TestPartialGauss(unittest.TestCase):
    def test_1_2_3_rows_reduce_k1(self):
        a = np.ones((3, 3))
        a[1, :] = 2
        a[2, :] = 3
        k = 1
        reduced_a = partial_row_reduce(a, k)
        expected_a = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        self.assertTrue(np.all(reduced_a == expected_a))

    def test_incremental_rows_reduce_k1(self):
        a = np.arange(1, 10).reshape(3, 3)
        k = 1
        reduced_a = partial_row_reduce(a, k)
        expected_a = np.array([[1, 2, 3], [0, -3, -6], [0, -6, -12]])
        self.assertTrue(np.all(reduced_a == expected_a))

    def test_incremental_rows_reduce_k2(self):
        a = np.arange(1, 10).reshape(3, 3)
        k = 2
        reduced_a = partial_row_reduce(a, k)
        expected_a = np.array([[1, 2, 3], [0, -3, -6], [0, 0, 0]])
        self.assertTrue(np.all(reduced_a == expected_a))


if __name__ == '__main__':
    unittest.main()
