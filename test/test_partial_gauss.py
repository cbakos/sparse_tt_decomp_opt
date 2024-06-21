import unittest
import numpy as np
import scipy as sp
import sympy
from scipy.io import mmread
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu

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

    def compute_nonzero_entries_of_reduced_systems(self, matrix_name: str, threshold: float = 1e-7):
        # Q2: why don't we get that U (from LU factorization) or L from Cholesky equal row-reduced matrix?
        # - probably because of pivoting/ variable reordering of library methods
        path = "../data/{}/{}.mtx".format(matrix_name, matrix_name)
        a = mmread(path)
        a = a.toarray()

        b = np.sum(a, axis=1).reshape(-1, 1).astype(np.float64)
        n = a.shape[0]
        k = n
        ab = np.hstack((a, b))

        # complete reduction
        reduced_ab = partial_row_reduce(ab, k)

        slu = splu(a, permc_spec="NATURAL", diag_pivot_thresh=0, options={"SymmetricMode": True})
        u = slu.U.toarray()
        l = slu.L.toarray()

        # round entries close to zero - for both to ensure nnz counts match
        reduced_ab = np.where(np.abs(reduced_ab) < threshold, 0, reduced_ab)
        u = np.where(np.abs(u) < threshold, 0, u)

        reduced_a = reduced_ab[:, :n]

        # check the lu-solve gives right result
        y = sp.linalg.solve(l, b)
        x = sp.linalg.solve(u, y)
        self.assertTrue(np.allclose(x, np.ones_like(x)))

        # check that nonzero entries match for partial Gauss and U from LU decomposition, up to the threshold
        self.assertTrue(np.allclose(u, reduced_a, atol=threshold))

        # also check number of nonzero entries
        reduced_a_nnz = np.count_nonzero(reduced_a)
        u_nnz = np.count_nonzero(u)
        self.assertEqual(reduced_a_nnz, u_nnz)

    def test_numerical_error_nonzero_count_ex5(self):
        self.compute_nonzero_entries_of_reduced_systems("ex5")

    def test_numerical_error_nonzero_count_ex3(self):
        self.compute_nonzero_entries_of_reduced_systems("ex3")

    def test_numerical_error_nonzero_count_ex10(self):
        self.compute_nonzero_entries_of_reduced_systems("ex10")

    def test_numerical_error_nonzero_count_ex10hs(self):
        self.compute_nonzero_entries_of_reduced_systems("ex10hs")

    def test_numerical_error_nonzero_count_ex13(self):
        self.compute_nonzero_entries_of_reduced_systems("ex13")

    # def test_numerical_error_nonzero_count_ex15(self):
    #     self.compute_nonzero_entries_of_reduced_systems("ex15")

    def test_numerical_error_nonzero_count_bcsstk13(self):
        self.compute_nonzero_entries_of_reduced_systems("bcsstk13")

    # def test_numerical_error_nonzero_count_Pres_Poisson(self):
    #     self.compute_nonzero_entries_of_reduced_systems("Pres_Poisson")

    # Q: does partial gauss get wrong results due to numerical errors?
    # A: for smaller matrices not the case, see the test cases below. For ex13, we get the most numerical errors but
    # still get correct solution up to 1e-3 (absolute error).
    def complete_gauss_elimination_per_matrix(self, matrix_name: str, threshold: float = 1e-10):
        path = "../data/{}/{}.mtx".format(matrix_name, matrix_name)
        a = mmread(path)
        a = a.toarray()
        n = a.shape[0]
        b = np.sum(a, axis=1).reshape(-1, 1).astype(np.float64)
        k = n
        ab = np.hstack((a, b))

        # reduce part
        reduced_ab = partial_row_reduce(ab, k)

        # back substitution
        sol = partial_gauss_back_subst(reduced_ab[:, :n], np.zeros_like(b), reduced_ab[:, n:], k)

        expected_sol = np.ones_like(sol)
        self.assertTrue(np.allclose(sol, expected_sol, atol=threshold))

    def test_complete_gauss_elimination_ex5(self):
        self.complete_gauss_elimination_per_matrix("ex5")

    def test_complete_gauss_elimination_ex3(self):
        self.complete_gauss_elimination_per_matrix("ex3")

    def test_complete_gauss_elimination_ex10(self):
        self.complete_gauss_elimination_per_matrix("ex10")

    def test_complete_gauss_elimination_ex10hs(self):
        self.complete_gauss_elimination_per_matrix("ex10hs")

    def test_complete_gauss_elimination_ex13(self):
        self.complete_gauss_elimination_per_matrix("ex13", threshold=1e-3)

    def test_complete_gauss_elimination_bcsstk13(self):
        self.complete_gauss_elimination_per_matrix("bcsstk13")


if __name__ == '__main__':
    unittest.main()
