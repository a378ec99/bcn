"""Cost testing.

Notes
-----
Defines a class that assert the functioning of the `cost` module.
"""
from __future__ import division, absolute_import


__all__ = ['TestCost']

import sys
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

import unittest
import numpy as np
from bcn.data import DataSimulated
from bcn.cost import Cost
from bcn.linear_operators import LinearOperatorEntry, LinearOperatorDense, LinearOperatorKsparse, LinearOperatorCustom
from bcn.utils.testing import assert_consistency


class TestCost(unittest.TestCase):
    """Test to verify that the cost function produces the correct result..

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (50, 60)
        Shape of the output bias matrix in the form of (n_samples, n_features).
    rank : int, default = 6
        Rank of the low-rank decomposition of the bias matrix.
    n_measurements : int (default = 901)
        Number of linear operators and measurements to be generated.
    data : Data object, default = None
        Generated on the fly by the `run` method.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.rank = 2
        self.n_measurements = 1000
        self.data = None
    
    def test_entry(self):
        self.data = DataSimulated(self.shape, self.rank)
        operator = LinearOperatorEntry(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        error_zero = cost.cost_func(self.data.d['sample']['true_bias'])
        assert error_zero == 0.0
        error_nonzero = cost.cost_func(self.data.d['sample']['true_bias'] + np.random.normal(0, 1, self.data.d['sample']['mixed'].shape))
        assert error_nonzero != 0.0
        assert_consistency(error_nonzero, '825927bc02df4562bcb901896dad7ad7')
    
    def test_dense(self):
        self.data = DataSimulated(self.shape, self.rank)
        operator = LinearOperatorDense(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        error_zero = cost.cost_func(self.data.d['sample']['true_bias'])
        assert error_zero == 0.0
        error_nonzero = cost.cost_func(self.data.d['sample']['true_bias'] + np.random.normal(0, 1, self.data.d['sample']['mixed'].shape))
        assert error_nonzero != 0.0
        assert_consistency(error_nonzero, '7a98003d7deb2570abbb4d846def4270')

    def test_ksparse(self):
        self.data = DataSimulated(self.shape, self.rank)
        sparsity = 2
        operator = LinearOperatorKsparse(self.data, self.n_measurements, sparsity).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        error_zero = cost.cost_func(self.data.d['sample']['true_bias'])
        assert error_zero == 0.0
        error_nonzero = cost.cost_func(self.data.d['sample']['true_bias'] + np.random.normal(0, 1, self.data.d['sample']['mixed'].shape))
        assert error_nonzero != 0.0
        assert_consistency(error_nonzero, '270677a6cbe716139280db670b1774a5')
        
    def test_blind(self):
        self.data = DataSimulated(self.shape, self.rank)
        self.data.estimate()
        operator = LinearOperatorCustom(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        
        error_zero = cost.cost_func(self.data.d['sample']['true_bias']) # NOTE Can't be exactly zero because estimating everything... need to check the faults!
        print '---------------- COST:', error_zero
        #assert np.isclose(error_zero, 0.0, rtol=1e-5, atol=1e-5)
        
        error_nonzero = cost.cost_func(self.data.d['sample']['true_bias'] + np.random.normal(0, 1, self.data.d['sample']['mixed'].shape))
        assert error_nonzero != 0.0
        print '---------------- BAD COST:', error_nonzero
        assert_consistency(error_nonzero, '1ff66eb670796e29c48d2e68588d8a2b')

        #error_guess = cost.cost_func(self.data.d['sample']['guess_X'])
        #print '---------------- GUESS COST:', error_guess

        error_zeros = cost.cost_func(np.zeros_like(self.data.d['sample']['true_bias']))
        print '---------------- ZERO COST:', error_zeros


if __name__ == '__main__':
    unittest.main()