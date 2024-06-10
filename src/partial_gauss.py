from typing import Tuple

import sympy.matrices as sm
import numpy as np


def partial_row_reduce(a: np.array, k: int) -> np.array:
    """
    Performs partial Gauss elimination on a dense matrix a with right-hand side b for the first k variables.
    :param a: dense coefficient matrix
    :param b: right hand side vector
    :param k: first k number of variables to reduce
    :return: a, b; the partially row-reduced versions of a and b
    """
    # Determine n
    n = a.shape[0]

    # Elimination process
    for i in range(k):  # first k rows
        # Elementary row operations
        a_i = a[i, :]
        a_ii = a_i[i]
        for j in range(i + 1, n):  # columns from pivot column i
            m = a[j, i] / a_ii
            a[j, :] = a[j, :] - m * a_i

    return a


def partial_gauss_back_subst(a, k, x):
    n = a.shape[0]
    s = 0
    for i in range(k - 1, -1, -1):  # rows from k-1 down to 0
        for j in range(i + 1, n):  # columns from i+1 to n-1
            s += a[i, j] * x[j]
        x[i] = (a[i, n] - s) / a[i, i]
        s = 0
    return x


# https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrixbase.MatrixBase.elementary_row_op
