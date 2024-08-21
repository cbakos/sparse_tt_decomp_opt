import numpy as np
import scipy.sparse as ssp


def sparse_padding(a: ssp.coo_array, k: int) -> ssp.csr_matrix:
    """
    Adds k rows/columns of zero padding to a sparse matrix with ones on the extended diagonal.
    :param a: sparse matrix
    :param k: number of zero rows/columns to add
    :return: padded sparse matrix
    """
    # get shapes
    new_shape = np.array(a.shape) + k
    old_z = len(a.data)
    old_n = a.shape[0]

    # assign values for extension
    values = np.ones(old_z + k)
    rows = np.zeros(old_z + k)
    cols = np.zeros(old_z + k)

    # use correct indices for new values
    rows[old_z:] = np.arange(old_n, old_n + k)
    cols[old_z:] = np.arange(old_n, old_n + k)

    # use old values for original part
    rows[:old_z] = a.row
    cols[:old_z] = a.col
    values[:old_z] = a.data

    padded_a = ssp.csr_matrix((values, (rows, cols)), shape=new_shape)
    return padded_a
