from typing import Tuple

import cvxopt
import numpy as np
import scipy.sparse as ssp

from optimizers.padding import sparse_padding
from optimizers.partial_gauss import partial_row_reduce, partial_row_reduce_step
from optimizers.variable_ordering import rcm_order


def amd_module(a: ssp.csr_matrix) -> ssp.csr_matrix:
    a = a.tocoo()
    spa = cvxopt.spmatrix(a.data, a.row, a.col, size=a.shape)
    order = cvxopt.amd.order(spa)
    # perform row and column permutations as described in: https://cvxopt.org/userguide/spsolvers.html#matrix-orderings
    spa_reordered = spa[order, order]

    # get coo values of reordered sparse matrix
    values = np.array(spa_reordered.V).flatten()
    rows = np.array(spa_reordered.I).flatten()
    cols = np.array(spa_reordered.J).flatten()

    # construct result in scipy sparse matrix format
    a = ssp.coo_matrix((values, (rows, cols)), shape=spa_reordered.size)
    a = a.tocsr()
    return a


def partial_gauss_module(a: ssp.csr_matrix, num_variables: int, threshold: float = 1e-7) \
        -> Tuple[ssp.csr_matrix, int, int]:
    a = a.toarray()
    full_a = partial_row_reduce(a, num_variables)
    # Use np.where to set values close to zero, to zero
    a = np.where(np.abs(a) < threshold, 0, a)

    # count nonzero entries of full (partially row-reduced) matrix
    z_full = np.count_nonzero(a)

    # get remaining part
    a = full_a[num_variables:, num_variables:]

    # number of nonzero entries in sub-matrix
    z_reduced = np.count_nonzero(a)

    a = ssp.csr_matrix(a)
    return a, z_full, z_reduced


def padding_module(a: ssp.csr_matrix, num_variables: int) -> ssp.csr_matrix:
    a = sparse_padding(a.tocoo(), num_variables)
    return a


def rcm_module(a: ssp.csr_matrix) -> ssp.csr_matrix:
    # get rcm order
    order = rcm_order(a)
    order = cvxopt.matrix(order)

    a = a.tocoo()
    spa = cvxopt.spmatrix(a.data, a.row, a.col, size=a.shape)
    # perform row and column permutations as described in: https://cvxopt.org/userguide/spsolvers.html#matrix-orderings
    spa_reordered = spa[order, order]

    # get coo values of reordered sparse matrix
    values = np.array(spa_reordered.V).flatten()
    rows = np.array(spa_reordered.I).flatten()
    cols = np.array(spa_reordered.J).flatten()

    # construct result in scipy sparse matrix format
    a = ssp.coo_matrix((values, (rows, cols)), shape=spa.size)
    a = a.tocsr()
    return a
