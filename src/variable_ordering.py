import numpy as np
import scipy.sparse as ssp
import cvxopt
from cvxopt import amd
from scipy.sparse.csgraph import reverse_cuthill_mckee


def amd_order(a: ssp.spmatrix) -> np.array:
    """
    Takes a sparse, square scipy matrix and returns its approximate minimum degree (amd) variable ordering.
    :param a: the sparse matrix
    :return: amd variable ordering as (n,) np array
    """
    # convert to cvxopt sparse matrix format
    spa = cvxopt.spmatrix(a.data, a.row, a.col)
    amd_order = amd.order(spa)
    order = np.array(amd_order).flatten()  # get order as np array
    return order


def rcm_order(a: ssp.spmatrix, is_symmetric: bool = True) -> np.array:
    """
    Takes a sparse, square scipy matrix and returns its reverse Cuthill-McKee variable ordering.
    :param is_symmetric: whether matrix is symmetric
    :param a: the sparse matrix
    :return: rcm variable ordering as (n,) np array
    """
    return reverse_cuthill_mckee(a.tocsr(), symmetric_mode=is_symmetric)
