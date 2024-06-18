import unittest

import cvxopt
import numpy as np
from scipy.io import mmread

from experiments.experiment_modules import partial_gauss_module, amd_module, rcm_module
from optimizers.variable_ordering import amd_order, rcm_order


class TestExperimentModules(unittest.TestCase):
    def test_amd_module(self):
        matrix_name = "ex5"
        path = "../data/{}/{}.mtx".format(matrix_name, matrix_name)
        a = mmread(path)

        # manual ordering
        order = amd_order(a)
        a_dense = a.toarray()
        a_dense[:] = a_dense[:, order]
        a_dense[:] = a_dense[order, :]

        # amd_module ordering
        a_amd = amd_module(a.tocsr())
        a_amd = a_amd.toarray()
        self.assertTrue(np.allclose(a_dense, a_amd))

    def test_rcm_module(self):
        matrix_name = "ex5"
        path = "../data/{}/{}.mtx".format(matrix_name, matrix_name)
        a = mmread(path)

        # manual ordering
        order = rcm_order(a)
        a_dense = a.toarray()
        a_dense[:] = a_dense[:, order]
        a_dense[:] = a_dense[order, :]

        # rcm_module ordering
        a_rcm = rcm_module(a.tocsr())
        a_rcm = a_rcm.toarray()
        self.assertTrue(np.allclose(a_dense, a_rcm))


if __name__ == '__main__':
    unittest.main()
