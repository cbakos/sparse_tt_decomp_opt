import unittest

import numpy as np
from scipy.sparse import csr_matrix

from optimizers.tile_size import get_rank_from_tile_size


class TestTileSize(unittest.TestCase):
    def test_get_rank_from_tile_size_one_block(self):
        a = np.zeros((10, 10))
        a[:5, :5] = 1
        tile_size = 5
        a = csr_matrix(a)
        r, rows = get_rank_from_tile_size(a, tile_size)
        expected_r = 1
        col_0 = np.array([0])
        col_1 = np.array([])
        self.assertEqual(r, expected_r)
        self.assertTrue(np.all(col_0 == rows[0]))
        self.assertTrue(np.all(col_1 == rows[1]))

    def test_get_rank_from_tile_size_block_diag(self):
        a = np.zeros((10, 10))
        a[:5, :5] = 1
        a[5:10, 5:10] = 1
        tile_size = 5
        a = csr_matrix(a)
        r, rows = get_rank_from_tile_size(a, tile_size)
        expected_r = 2
        col_0 = np.array([0])
        col_1 = np.array([1])
        self.assertEqual(r, expected_r)
        self.assertTrue(np.all(col_0 == rows[0]))
        self.assertTrue(np.all(col_1 == rows[1]))

    def test_get_rank_from_tile_size_diag(self):
        a = np.diag(np.arange(1, 7))
        tile_size = 2
        a = csr_matrix(a)
        r, rows = get_rank_from_tile_size(a, tile_size)
        expected_r = 3
        col_0 = np.array([0])
        col_1 = np.array([1])
        col_2 = np.array([2])
        self.assertEqual(r, expected_r)
        self.assertTrue(np.all(col_0 == rows[0]))
        self.assertTrue(np.all(col_1 == rows[1]))
        self.assertTrue(np.all(col_2 == rows[2]))

    def test_get_rank_from_tile_size_other_diag(self):
        a = np.zeros((6, 6))
        a[5, 0] = 1
        a[4, 1] = 2
        a[3, 2] = 3
        a[2, 3] = 4
        a[1, 4] = 5
        a[0, 5] = 6
        tile_size = 2
        a = csr_matrix(a)
        r, rows = get_rank_from_tile_size(a, tile_size)
        expected_r = 3
        col_0 = np.array([2])
        col_1 = np.array([1])
        col_2 = np.array([0])
        self.assertEqual(r, expected_r)
        self.assertTrue(np.all(col_0 == rows[0]))
        self.assertTrue(np.all(col_1 == rows[1]))
        self.assertTrue(np.all(col_2 == rows[2]))

    def test_get_rank_from_tile_size_4_tiles(self):
        a = np.zeros((6, 6))
        a[0, 0] = 1
        a[2, 0] = 2
        a[4, 0] = 3
        a[0, 2] = 4
        tile_size = 2
        a = csr_matrix(a)
        r, rows = get_rank_from_tile_size(a, tile_size)
        expected_r = 4
        col_0 = np.array([0, 1, 2])
        col_1 = np.array([0])
        col_2 = np.array([])
        self.assertEqual(r, expected_r)
        self.assertTrue(np.all(col_0 == rows[0]))
        self.assertTrue(np.all(col_1 == rows[1]))
        self.assertTrue(np.all(col_2 == rows[2]))


if __name__ == '__main__':
    unittest.main()
