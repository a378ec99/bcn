"""Visualization testing module.

Notes
-----
This module define a classes that tests the functioning of the `visualization` module and its components.
"""
from __future__ import division, absolute_import


__all__ = ['TestDependences', 'TestCorrelations']

import unittest
import hashlib
import sys
import numpy as np
sys.path.append('/home/sohse/projects/PUBLICATION/GIT2/bcn')
from data import DataSimulated, DataSimulatedLarge
from visualization import visualize_dependences, visualize_correlations
from cost import Cost
from solvers import ConjugateGradientSolver
from linear_operators import LinearOperatorBlind


def _assert_consistency(X, true_md5):
    m = hashlib.md5()
    m.update(X)
    current_md5 = m.hexdigest()
    assert current_md5 == true_md5

    
class TestDependences(unittest.TestCase):
    """Tests to verify that the visualizations produced without errors for the blind recovery approach.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (30, 30)
        Shape of the output bias matrix in the form of (n_samples, n_features).
    rank : int, default = 2
        Rank of the low-rank decomposition.
    n_measurements: int, default = 1000
        Number of linear operators and measurements to be generated.
    data : dict, default = None
        Data contained which is generated on the fly.

    Note
    ----
    The visualizations are not checked for actual correctness, only their construction for errors.
    """
    
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = None
        self.rank = 2
        self.n_measurements = 5000
        self.data = None
    '''
    def test_dependences_true(self):
        self.shape = (50, 60)
        self.data = DataSimulated(self.shape, self.rank)
        self.data.estimate(true_stds={'sample': self.data.d['sample']['true_stds'], 'feature': self.data.d['feature']['true_stds']}, true_pairs={'sample': self.data.d['sample']['true_pairs'], 'feature': self.data.d['feature']['true_pairs']}, true_directions={'sample': self.data.d['sample']['true_directions'], 'feature': self.data.d['feature']['true_directions']})
        operator = LinearOperatorBlind(self.data.d, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_function, self.data, self.rank, 1, verbosity=0)
        self.data.d = solver.recover()
        visualize_dependences(self.data.d, file_name='test_dependences_true')
        #_assert_consistency(self.data.d['sample']['estimated_bias'], 1)
    '''    
    def test_dependences_true_large(self):
        self.shape = (150, 160)
        self.data = DataSimulatedLarge(self.shape, self.rank, subset_factor=1)
        self.data.estimate(true_stds={'sample': self.data.d['sample']['true_stds'], 'feature': self.data.d['feature']['true_stds']}, true_pairs={'sample': self.data.d['sample']['true_pairs'], 'feature': self.data.d['feature']['true_pairs']}, true_directions={'sample': self.data.d['sample']['true_directions'], 'feature': self.data.d['feature']['true_directions']})
        operator = LinearOperatorBlind(self.data.d, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_function, self.data, self.rank, 10, verbosity=0)
        self.data.d = solver.recover()
        visualize_dependences(self.data.d, file_name='test_dependences_true_large')
        #_assert_consistency(self.data.d['sample']['estimated_bias'], 1)
    '''  
    def test_dependences_estimated(self):
        self.shape = (50, 60)
        self.data = DataSimulated(self.shape, self.rank)
        self.data.estimate()
        operator = LinearOperatorBlind(self.data.d, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_function, self.data, self.rank, 1, verbosity=0)
        self.data.d = solver.recover()
        visualize_dependences(self.data.d, file_name='test_dependences_estimated')
        #_assert_consistency(self.data.d['sample']['estimated_bias'], 1)
    '''     
    def test_dependences_estimated_large(self):
        # NOTE Only estimates standard deviations, pairs and directions are given.
        self.shape = (150, 160)
        self.data = DataSimulatedLarge(self.shape, self.rank)
        self.data.estimate(true_pairs={'sample': self.data.d['sample']['true_pairs'], 'feature': self.data.d['feature']['true_pairs']}, true_directions={'sample': self.data.d['sample']['true_directions'], 'feature': self.data.d['feature']['true_directions']})
        operator = LinearOperatorBlind(self.data.d, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_function, self.data, self.rank, 1, verbosity=0)
        self.data.d = solver.recover()
        visualize_dependences(self.data.d, file_name='test_dependences_estimated_large')
        #_assert_consistency(self.data.d['sample']['estimated_bias'], 1)
    
                
class TestCorrelations(unittest.TestCase):
    """Tests to verify that the visualizations produced without errors for the blind recovery approach.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (30, 30)
        Shape of the output bias matrix in the form of (n_samples, n_features).
    rank : int, default = 2
        Rank of the low-rank decomposition.
    n_measurements: int, default = 1000
        Number of linear operators and measurements to be generated.
    data : dict
        Data contained which is generated by simulation.

    Note
    ----
    The visualizations are not checked for actual correctness, only their construction for errors.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (30, 30)
        self.rank = 2
        self.data = DataSimulated(self.shape, self.rank)
    
    def test_correlations(self):
        visualize_correlations(self.data.d)
        #_assert_consistency(self.data.d['sample']['estimated_correlations'], '5e1a8a1db319261f0cf28e9292cfc812')

    
if __name__ == '__main__':
    unittest.main()






    