import cvxopt
import numpy as np
import scipy.sparse as ssp

from optimizers.padding import sparse_padding
from optimizers.partial_gauss import partial_row_reduce
from optimizers.variable_ordering import amd_order, rcm_order


def amd_module(a: ssp.csr_matrix) -> ssp.csr_matrix:
    a = a.tocoo()
    spa = cvxopt.spmatrix(a.data, a.row, a.col)
    order = cvxopt.amd.order(spa)
    # perform row and column permutations
    spa_reordered = spa[order, order]
    values = np.array(spa_reordered.V).flatten()
    rows = np.array(spa_reordered.I).flatten()
    cols = np.array(spa_reordered.J).flatten()
    a = ssp.coo_matrix((values, (rows, cols)))
    a = a.tocsr()
    return a


def partial_gauss_module(a: ssp.csr_matrix, num_variables: int, threshold: float = 1e-7) -> ssp.csr_matrix:
    a = a.toarray()
    full_a = partial_row_reduce(a, num_variables)
    # get remaining part
    a = full_a[num_variables:, num_variables:]
    # Use np.where to set values close to zero, to zero
    a = np.where(np.abs(a) < threshold, 0, a)
    a = ssp.csr_matrix(a)
    return a


def padding_module(a: ssp.csr_matrix, num_variables: int) -> ssp.csr_matrix:
    a = sparse_padding(a.tocoo(), num_variables)
    return a


def rcm_module(a: ssp.csr_matrix) -> ssp.csr_matrix:
    order = rcm_order(a)
    a = a.tocsr()
    a.indices = order.take(a.indices)
    a = a.tocsc()
    a.indices = order.take(a.indices)
    return a
