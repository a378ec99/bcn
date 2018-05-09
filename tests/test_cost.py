"""Cost testing.

Notes
-----
Tests the functioning of the `cost` module.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

#from bcn.data import DataSimulated
from bcn.cost import Cost


class TestSimple(unittest.TestCase):

    def setUp(self):
        self.sparsity = 1
        self.X = np.ones((10, 9))
        self.X[0, :] = 2.0
        self.y = [0.0, 4.0]
        self.A = [{'row': [0], 'col': [8], 'value': [3.5]},
                  {'row': [9], 'col': [7], 'value': [4.5]}]
       
    def test_entry(self):
        cost = Cost(self.A, self.y, self.sparsity)
        expected = np.mean([(0.0 - self.X[0, 8] * 3.5)**2, (4.0 - self.X[9, 7] * 4.5)**2])
        assert cost.cost_func(self.X) == expected
       
    def test_ksparse(self):
        sparsity = 2 
        A = [{'row': [0, 4], 'col': [8, 4], 'value': [3.5, 1.5]},
             {'row': [9, 5], 'col': [7, 8], 'value': [4.5, 0.9]}]
        cost = Cost(A, self.y, sparsity)
        expected = np.mean([(0.0 - (self.X[0, 8] * 3.5 + self.X[4, 4] * 1.5))**2, (4.0 - (self.X[9, 7] * 4.5 + self.X[5, 8] * 0.9))**2])
        assert cost.cost_func(self.X) == expected

    def test_dense(self):
        sparsity = self.X.shape[0] * self.X.shape[1]
        row, col = np.indices(self.X.shape)
        A = [{'row': list(row.ravel()), 'col': list(col.ravel()), 'value': range(90)}]
        y = list(np.ones(90) * 1.4)
        cost = Cost(A, y, sparsity)
        A_matrix = np.reshape(range(90), (10, 9))
        expected = np.mean((y - np.sum([A_matrix * self.X]))**2)
        assert cost.cost_func(self.X) == expected
   
    def test_X_nan(self):
        X = np.array(self.X)
        X[0, :] = np.nan
        cost = Cost(self.A, self.y, self.sparsity)
        with self.assertRaises(AssertionError):
            cost.cost_func(X)
            
    def test_zero_A(self):
        A_zero = [{'row': [0], 'col': [8], 'value': [0.0]},
                  {'row': [9], 'col': [7], 'value': [0.0]}]
        cost = Cost(A_zero, self.y, self.sparsity)
        expected = np.mean([0.0, (4.0 - 0.0)**2])
        assert cost.cost_func(self.X) == expected
        
    def test_zero_X(self):
        X_zero = np.zeros((10, 9))
        cost = Cost(self.A, self.y, self.sparsity)
        expected = np.mean([0.0, (4.0 - 0.0)**2])
        assert cost.cost_func(X_zero) == expected
    
'''    
class TestUseCase(unittest.TestCase):
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
'''

if __name__ == '__main__':
    unittest.main()
