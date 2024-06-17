import unittest

import cvxopt
import numpy as np
from scipy.io import mmread

from experiments.experiment_modules import partial_gauss_module, amd_module
from optimizers.variable_ordering import amd_order


class TestExperimentModules(unittest.TestCase):
    def test_amd_module(self):
        matrix_name = "ex5"
        path = "../data/{}/{}.mtx".format(matrix_name, matrix_name)
        a = mmread(path)
        spa = cvxopt.spmatrix(a.data, a.row, a.col)
        order = cvxopt.amd.order(spa)
        spa_reordered = spa[order, order]
        spa = np.zeros((spa_reordered.size[0], spa_reordered.size[1]))

        # Fill the dense matrix with values from the sparse matrix
        for k in range(len(spa_reordered.I)):
            i = spa_reordered.I[k]
            j = spa_reordered.J[k]
            v = spa_reordered.V[k]
            spa[i, j] = v

        a_dense = a.toarray()
        order = np.array(order).flatten()
        idx = np.empty_like(order)
        idx[order] = np.arange(len(order))
        a_dense[:] = a_dense[:, order]
        a_dense[:] = a_dense[order, :]

        a_amd = amd_module(a.tocsr())
        a_amd = a_amd.toarray()
        self.assertTrue(np.allclose(a_dense, a_amd))
        self.assertTrue(np.allclose(spa, a_dense))


if __name__ == '__main__':
    unittest.main()
