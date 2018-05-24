"""Cost function tests.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.cost import Cost


class TestSimple(unittest.TestCase):
    """Test to verify that all three models produce finite ndarrays.
    """
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
    

if __name__ == '__main__':
    unittest.main()
