import numpy as np
from scipy.io import mmread
import numpy.linalg as nla
import scipy.linalg as sla
import scikit_tt as stt

from optimizers.tile_size import prime_factors


def get_matrix(matrix_name: str, n: int = 2000) -> np.ndarray:
    x = np.arange(n)
    if matrix_name == "hilbert":
        dense_a = sla.hilbert(n)
    elif matrix_name == "vandermonde":
        dense_a = np.vander(x)
    elif matrix_name == "hankel":
        dense_a = sla.hankel(x)
    elif matrix_name == "toeplitz":
        dense_a = sla.toeplitz(x)
    else:
        path = "../../data/{}/{}.mtx".format(matrix_name, matrix_name)
        a = mmread(path)  # reads to coo_matrix format
        dense_a = a.toarray()
    return dense_a


def get_ranks_for_matrix(matrix_name: str):
    dense_a = get_matrix(matrix_name)
    tol1 = 1e-3
    tol2 = 1e-10
    print("matrix: {}".format(matrix_name))
    print("rank: ", nla.matrix_rank(dense_a))
    print("numerical rank, tol {}: {}".format(tol1, nla.matrix_rank(dense_a, tol=tol1)))
    print("numerical rank, tol {}: {}".format(tol2, nla.matrix_rank(dense_a, tol=tol2)))

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
    matrix_names = ["ex10", "ex3", "ex10hs", "ex13", "hilbert", "vandermonde", "hankel", "toeplitz"]
    for matrix_name in matrix_names:
        get_ranks_for_matrix(matrix_name)
