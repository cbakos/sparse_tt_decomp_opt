import numpy as np
from scipy.io import mmread
import cvxopt
from cvxopt import amd

from variable_ordering import amd_order, rcm_order

if __name__ == '__main__':
    path = "../data/ex10/ex10.mtx"
    a = mmread(path)
    order1 = amd_order(a)
    order2 = rcm_order(a)
    order2
