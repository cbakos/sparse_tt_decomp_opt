import numpy as np
from scipy.io import mmread


def is_wdd(a: np.array) -> bool:
    """
    Determines whether a numpy array is weakly diagonally dominant or not.
    :param a:
    :return:
    """
    d = np.diag(np.abs(a))  # Find diagonal coefficients
    s = np.sum(np.abs(a), axis=1) - d  # Find row sum without diagonal
    if np.all(d >= s):
        return True
    else:
        return False


if __name__ == '__main__':
    matrix_names = ["ex3", "ex10", "ex10hs", "ex13", "ex15", "Pres_Poisson", "bcsstk13"]
    for matrix_name in matrix_names:
        path = "../../data/{}/{}.mtx".format(matrix_name, matrix_name)
        a = mmread(path)
        a = a.toarray()
        print(matrix_name, is_wdd(a))
