import numpy as np
from scipy.io import mmread
import numpy.linalg as la
import scikit_tt as stt

from optimizers.tile_size import prime_factors


def get_ranks_for_matrix(matrix_name: str):
    path = "../../data/{}/{}.mtx".format(matrix_name, matrix_name)
    a = mmread(path)  # reads to coo_matrix format
    dense_a = a.toarray()
    tol1 = 1e-3
    tol2 = 1e-10
    print("matrix: {}".format(matrix_name))
    print("rank: ", la.matrix_rank(dense_a))
    print("numerical rank, tol {}: {}".format(tol1, la.matrix_rank(dense_a, tol=tol1)))
    print("numerical rank, tol {}: {}".format(tol2, la.matrix_rank(dense_a, tol=tol2)))

    mode_sizes = prime_factors(dense_a.shape[0])
    mode_sizes = mode_sizes + mode_sizes
    dense_a = np.reshape(dense_a, mode_sizes)
    a_tt = stt.TT(dense_a, threshold=0)
    print("TT-rank, tol=0: ", max(a_tt.ranks))

    a_tt = stt.TT(dense_a, threshold=tol1)
    print("TT-rank, tol {}: {}".format(tol1, max(a_tt.ranks)))

    a_tt = stt.TT(dense_a, threshold=tol2)
    print("TT-rank, tol {}: {}".format(tol2, max(a_tt.ranks)))
    print("____")


if __name__ == '__main__':
    matrix_names = ["ex10", "ex3", "ex10hs", "ex13"]
    for matrix_name in matrix_names:
        get_ranks_for_matrix(matrix_name)
