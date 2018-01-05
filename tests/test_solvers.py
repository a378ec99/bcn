"""Solvers testing.

Notes
-----
Defines a class that assert the functioning of the `solvers` module. Currently only supports testing ConjugateGradient class.
"""
from __future__ import division, absolute_import


__all__ = ['TestConjugateGradientSolver']

import sys
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

import unittest
import numpy as np
from bcn.data import DataSimulated
from bcn.cost import Cost
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorEntry, LinearOperatorDense, LinearOperatorKsparse, LinearOperatorCustom
from bcn.bias import guess_func
from bcn.utils.testing import assert_consistency


class TestConjugateGradientSolver(unittest.TestCase):
    """Test to verify that the conjugate gradient based solver produces the correct result.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (50, 60)
        Shape of the output bias matrix in the form of (n_samples, n_features).
    rank : int, default = 2
        Rank of the low-rank decomposition of the bias matrix.
    n_measurements : int, default = 1000
        Number of linear operators and measurements to be generated.
    data : Data object, default = None
        Generated on the fly by the `run` method.
        
    Note
    ----
    True pairs, directions and standard deviations are used for the estimates to focus only on the solver.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        self.shape = (30, 50)
        self.rank = 2
        self.n_measurements = 1000
        self.data = DataSimulated(self.shape, self.rank)
        
    def test_entry(self):
        operator = LinearOperatorEntry(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, self.rank, 1, verbosity=0)
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']
        
        print '-------------------------------------------------------- entry'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
        assert_consistency(X_estimated, 'a4ccf2bc272d6b4c3a4d797c6356e61d')

    def test_dense(self):
        operator = LinearOperatorDense(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, self.rank, 1, verbosity=0)
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']
        
        print '-------------------------------------------------------- dense'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
        assert_consistency(X_estimated, 'f082ec600a1b3188554758c4010061aa')
        
    def test_ksparse1(self):
        sparsity = 1
        operator = LinearOperatorKsparse(self.data, self.n_measurements, sparsity).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, self.rank, 1, verbosity=0)
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']
        
        print '-------------------------------------------------------- 1-sparse'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
        assert_consistency(X_estimated, '7418dbd052a2a4e00835dd9530c9199d')
        
    def test_ksparse2(self):
        sparsity = 2
        operator = LinearOperatorKsparse(self.data, self.n_measurements, sparsity).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, self.rank, 1, verbosity=0)
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']
        
        print '-------------------------------------------------------- 2-sparse'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
        assert_consistency(X_estimated, '06cc01673cf48472d34b83c08a08089d')
    
    def test_ksparse3(self):
        sparsity = 3
        operator = LinearOperatorKsparse(self.data, self.n_measurements, sparsity).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)

        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, self.rank, 1, verbosity=0)
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']
        
        print '-------------------------------------------------------- 3-sparse'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
        assert_consistency(X_estimated, '59bd166bfa201a1976d8d4011eb1449a')
        
    def test_blind(self):
        self.data.estimate(true_stds={'sample': self.data.d['sample']['true_stds'], 'feature': self.data.d['feature']['true_stds']}, true_pairs={'sample': self.data.d['sample']['true_pairs'], 'feature': self.data.d['feature']['true_pairs']}, true_directions={'sample': self.data.d['sample']['true_directions'], 'feature': self.data.d['feature']['true_directions']})
        operator = LinearOperatorCustom(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)

        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, self.rank, 3, verbosity=0)
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']
        
        print 'diff', X_estimated - X_true
        print '-------------------------------------------------------- blind'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
        assert_consistency(X_estimated, '41c05f45f27ca42dbc89186ecc0c7f22')

        
if __name__ == '__main__':
    unittest.main()









    