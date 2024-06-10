import numpy as np
from scipy.io import mmread
import cvxopt
from cvxopt import amd

from padding import sparse_padding
from tile_size import prime_factors, possible_tile_sizes_from_factors, get_rank_from_tile_size
from variable_ordering import amd_order, rcm_order

if __name__ == '__main__':
    matrix = "ex10"
    path = "../data/{}/{}.mtx".format(matrix, matrix)
    a = mmread(path)
    n = a.shape[0]
    z = a.nnz
    k = 3
    order1 = amd_order(a)
    order2 = rcm_order(a)

    padded_a = sparse_padding(a, k)

    factors = prime_factors(n)
    tiles = possible_tile_sizes_from_factors(factors)
    for tile in tiles:
        r, _ = get_rank_from_tile_size(a, tile)
        print(tile, r)

