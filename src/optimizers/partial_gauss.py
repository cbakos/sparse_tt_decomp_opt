import numpy as np


def partial_row_reduce(a: np.array, k: int) -> np.array:
    """
    Performs partial Gauss elimination on a dense matrix a with right-hand side b for the first k variables.
    :param a: dense coefficient matrix together with rhs b
    :param k: first k number of variables to reduce
    :return: a: the partially row-reduced versions of a and b
    """
    # Determine n
    n = a.shape[0]

    # Elimination process
    for i in range(k):  # first k rows
        a = partial_row_reduce_step(a, i, n)
    return a


def partial_row_reduce_step(a: np.array, i: int, n: int) -> np.array:
    """
    Performs a single row reduction step.
    :param a: matrix to reduce
    :param i: index of variable to reduce
    :param n: total number of variables == number of rows
    :return: a with ith variable reduced
    """
    a_i = a[i, :]
    a_ii = a_i[i]
    for j in range(i + 1, n):  # columns from pivot column i
        m = a[j, i] / a_ii
        a[j, :] = a[j, :] - m * a_i
    return a


def partial_gauss_back_subst(a, x, b, k):
    """
    Performs back substitution on a dense matrix a with partially solved solution vector x. Assumes that the last n-k
    variables are already solved.
    :param a: partially row-reduced matrix, last n-k variables are already solved (i.e. a equals identity for those)
    :param x: solution vector
    :param b: right hand side vector
    :param k: number of variables to solve
    :return: x, the solution vector
    """
    n = a.shape[0]
    s = 0
    for i in range(k - 1, -1, -1):  # rows from k-1 down to 0
        for j in range(i + 1, n):  # columns
            s += a[i, j] * x[j]
        x[i] = (b[i] - s) / a[i, i]
        s = 0
    return x
