import numpy as np
import scipy.sparse as ssp


def sparse_padding(a: ssp.spmatrix, k: int) -> ssp.spmatrix:
    """
    Adds k rows/columns of zero padding to a sparse matrix.
    :param a: sparse matrix
    :param k: number of zero rows/columns to add
    :return: padded sparse matrix
    """
    new_shape = np.array(a.shape) + k
    padded_a = ssp.coo_array((a.data, (a.row, a.col)), shape=new_shape)
    return padded_a

