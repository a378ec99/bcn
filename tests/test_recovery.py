"""Recovery testing.

Notes
-----
Defines a class that assert the functioning of the combined modules.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.data import DataSimulated
from bcn.cost import Cost
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorEntry, LinearOperatorDense, LinearOperatorKsparse, LinearOperatorCustom
from bcn.bias import guess_func
from bcn.utils.testing import assert_consistency
from bcn.utils.visualization import visualize_dependences, visualize_absolute


class TestRecoveryFullRank(unittest.TestCase):
    """Test to verify that the full rank recovery results in perfect correlations.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (50, 60)
        Shape of the output bias matrix in the form of (n_samples, n_features).
    rank : int, default = 2
        Rank of the low-rank decomposition.
    n_measurements : int, default = 1000
        Number of linear operators and measurements to be generated.
    data : Data object, default = None
        Generated on the fly by the `run` method.
        
    Note
    ----
    True pairs, directions and standard deviations are used for the estimates to focus only on the recovery routine and not the estimation.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        self.shape = (20, 20)
        self.rank = 2
        self.n_measurements = 300
        self.data = DataSimulated(self.shape, self.rank, correlation_strength=1.0, m_blocks_factor=self.shape[0] // 2, noise_amplitude=20)
        
    def test_rank2(self):
        self.data.estimate(true_stds={'sample': self.data.d['sample']['true_stds'], 'feature': self.data.d['feature']['true_stds']}, true_pairs={'sample': self.data.d['sample']['true_pairs'], 'feature': self.data.d['feature']['true_pairs']}, true_directions={'sample': self.data.d['sample']['true_directions'], 'feature': self.data.d['feature']['true_directions']})
        operator = LinearOperatorCustom(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)

        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, 2, 3, maxiter=1000, verbosity=2) # self.shape[0]
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']
        
        visualize_dependences(self.data, space='sample', file_name='../out/test_recovery2', truth_available=True, estimate_available=True, recovery_available=True, format='.png', max_plots=10, max_points=40)
        visualize_absolute(self.data, space='sample', file_name='../out/test_absolute2', format='.png', recovered=True)

    def test_rank10(self):
        self.data.estimate(true_stds={'sample': self.data.d['sample']['true_stds'], 'feature': self.data.d['feature']['true_stds']}, true_pairs={'sample': self.data.d['sample']['true_pairs'], 'feature': self.data.d['feature']['true_pairs']}, true_directions={'sample': self.data.d['sample']['true_directions'], 'feature': self.data.d['feature']['true_directions']})
        operator = LinearOperatorCustom(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        cost = Cost(A, y)

        solver = ConjugateGradientSolver(cost.cost_func, guess_func, self.data, 10, 3, maxiter=1000, verbosity=2)
        self.data = solver.recover()
        X_estimated = self.data.d['sample']['estimated_bias']
        X_true = self.data.d['sample']['true_bias']

        visualize_dependences(self.data, space='sample', file_name='../out/test_recovery10', truth_available=True, estimate_available=True, recovery_available=True, format='.png', max_plots=10, max_points=40)
        visualize_absolute(self.data, space='sample', file_name='../out/test_absolute10', format='.png', recovered=True)
        
if __name__ == '__main__':
    unittest.main()









    
