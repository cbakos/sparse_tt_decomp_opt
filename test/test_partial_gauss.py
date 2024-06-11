import unittest
import numpy as np
import sympy

from optimizers.partial_gauss import partial_row_reduce, partial_gauss_back_subst


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

    def test_random0_full_reduce(self):
        np.random.seed(0)
        a = np.random.randint(1, 100, (3, 3))
        k = 3
        reduced_a = partial_row_reduce(a, k)
        expected_a = sympy.Matrix(a).echelon_form()
        expected_a = np.array(expected_a).astype(np.float64)
        self.assertTrue(np.all(reduced_a == expected_a))

    def test_random1_full_reduce(self):
        np.random.seed(1)
        a = np.random.randint(1, 1000, (30, 30))
        k = 30
        reduced_a = partial_row_reduce(a, k)
        expected_a = sympy.Matrix(a).echelon_form()
        expected_a = np.array(expected_a).astype(np.float64)
        self.assertTrue(np.all(reduced_a == expected_a))

    def test_random2_full_reduce_n100(self):
        np.random.seed(2)
        a = np.random.randint(1, 10000, (100, 100))
        k = 100
        reduced_a = partial_row_reduce(a, k)
        expected_a = sympy.Matrix(a).echelon_form()
        expected_a = np.array(expected_a).astype(np.float64)
        self.assertTrue(np.all(reduced_a == expected_a))

    def test_random1_full_back_subst_4x4(self):
        n = 4
        np.random.seed(1)
        a = np.random.randint(1, n**2, (n, n)).astype(np.float64)
        b = np.sum(a, axis=1).reshape(-1, 1).astype(np.float64)
        k = n
        ab = np.hstack((a, b))

        # reduce part
        reduced_ab = partial_row_reduce(ab, k)

        # back substitution
        sol = partial_gauss_back_subst(reduced_ab[:, :n], np.zeros_like(b), reduced_ab[:, n:], k)
        expected_sol = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(sol, expected_sol))

    def test_random2_full_back_subst_10x10(self):
        n = 10
        np.random.seed(2)
        a = np.random.randint(1, n**2, (n, n)).astype(np.float64)
        b = np.sum(a, axis=1).reshape(-1, 1).astype(np.float64)
        k = n
        ab = np.hstack((a, b))

        # reduce part
        reduced_ab = partial_row_reduce(ab, k)

        # back substitution
        sol = partial_gauss_back_subst(reduced_ab[:, :n], np.zeros_like(b), reduced_ab[:, n:], k)
        expected_sol = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(sol, expected_sol))

    def test_sdd_partial_back_subst(self):
        n = 4
        np.random.seed(2)
        a = np.random.rand(n, n).astype(np.float64)
        a += np.diag(np.random.randint(n, n**2, n))
        b = np.sum(a, axis=1).reshape(-1, 1).astype(np.float64)
        k = 2
        ab = np.hstack((a, b))

        # reduce part
        reduced_ab = partial_row_reduce(ab, k)
        sub_solution = np.linalg.solve(ab[k:, k:n], ab[k:, n:])
        part_solved_a = reduced_ab[:, :n]
        part_solved_a[k:n, k:n] = np.eye(n-k, n-k)
        x = np.zeros_like(b)
        x[k:n] = sub_solution

        # back substitution
        sol = partial_gauss_back_subst(part_solved_a, x, reduced_ab[:, n:], k)
        expected_sol = np.ones((n, 1))
        self.assertTrue(np.allclose(sol, expected_sol))


if __name__ == '__main__':
    unittest.main()
