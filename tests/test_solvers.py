"""Solver tests.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.data import DataSimulated
from bcn.cost import Cost
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorEntry, LinearOperatorDense, LinearOperatorKsparse, LinearOperatorCustom
from bcn.bias import guess_func


class TestConjugateGradientSolver(unittest.TestCase):
    """Test to verify that the conjugate gradient based solver produces the correct result.
    """
    def setUp(self):
        np.random.seed(42)
        self.shape = (30, 50)
        self.rank = 2
        self.n_measurements = 1000
        self.data = DataSimulated(self.shape, self.rank)
        self.true_bias = self.data.d['sample']['true_bias']
        self.mixed = self.data.d['sample']['mixed']
        
    def test_entry(self):
        operator = LinearOperatorEntry(self.n_measurements).generate(self.true_bias)
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(self.mixed, cost.cost_func, guess_func, self.rank, verbosity=0)
        results = solver.recover()
        X_estimated = results['estimated_bias']
        X_true = self.true_bias
        print '-------------------------------------------------------- entry'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)

    def test_dense(self):
        operator = LinearOperatorDense(self.n_measurements).generate(self.true_bias)
        A = operator['A']
        y = operator['y']
        sparsity = self.mixed.size
        cost = Cost(A, y, sparsity)
        solver = ConjugateGradientSolver(self.mixed, cost.cost_func, guess_func, self.rank, verbosity=0)
        results = solver.recover()
        X_estimated = results['estimated_bias']
        X_true = self.true_bias
        print '-------------------------------------------------------- dense'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
        
    def test_ksparse1(self):
        sparsity = 1
        operator = LinearOperatorKsparse(self.n_measurements, sparsity).generate(self.true_bias)
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y, sparsity)
        solver = ConjugateGradientSolver(self.mixed, cost.cost_func, guess_func, self.rank, verbosity=0)
        results = solver.recover()
        X_estimated = results['estimated_bias']
        X_true = self.true_bias
        print '-------------------------------------------------------- 1-sparse'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)
    
    def test_ksparse2(self):
        sparsity = 2
        operator = LinearOperatorKsparse(self.n_measurements, sparsity).generate(self.true_bias)
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y, sparsity)
        solver = ConjugateGradientSolver(self.mixed, cost.cost_func, guess_func, self.rank, verbosity=0)
        results = solver.recover()
        X_estimated = results['estimated_bias']
        X_true = self.true_bias
        print '-------------------------------------------------------- 2-sparse'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)

    def test_ksparse3(self):
        sparsity = 3
        operator = LinearOperatorKsparse(self.n_measurements, sparsity).generate(self.true_bias)
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y, sparsity)
        solver = ConjugateGradientSolver(self.mixed, cost.cost_func, guess_func, self.rank, verbosity=0)
        results = solver.recover()
        X_estimated = results['estimated_bias']
        X_true = self.true_bias
        print '-------------------------------------------------------- 3-sparse'
        print 'cost.cost_func(X_estimated)', cost.cost_func(X_estimated)
        print 'cost.cost_func(X_true)', cost.cost_func(X_true)
        print '-------------------------------------------------------- '
        assert np.allclose(X_estimated, X_true, rtol=1e-5, atol=1e-5)

        
if __name__ == '__main__':
    unittest.main()









    
