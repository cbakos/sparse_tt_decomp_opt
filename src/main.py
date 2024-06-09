import numpy as np
from scipy.io import mmread
import cvxopt
from cvxopt import amd

from tile_size import prime_factors
from variable_ordering import amd_order, rcm_order

if __name__ == '__main__':
    matrix = "ex15"
    path = "../data/{}/{}.mtx".format(matrix, matrix)
    a = mmread(path)
    n = a.shape[0]
    z = a.nnz
    order1 = amd_order(a)
    order2 = rcm_order(a)

    factors = prime_factors(n)
    z
