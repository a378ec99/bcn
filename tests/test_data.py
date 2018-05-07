"""Data testing.

Notes
-----
Defines a test class that assert the functioning of the `data` module.
"""
from __future__ import division, absolute_import

import unittest

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl

from bcn.data import DataSimulated, DataBlind
from bcn.utils.testing import assert_consistency

        
class TestDataSimulated(unittest.TestCase):
    """Test to verify that the inital data is created correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, (default = (50, 60))
        Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features).
    rank : int, (default = 2)
        Rank of the low-rank decomposition.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.rank = 2
        
    def test(self):
        data = DataSimulated(shape=self.shape, rank=self.rank)
        data.estimate() # true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}
        for space in ['feature', 'sample']:
            assert data.d[space]['estimated_correlations'].shape == (data.d[space]['shape'][0], data.d[space]['shape'][0])
            assert data.d[space]['estimated_pairs'].shape[1] == 2
            assert data.d[space]['estimated_directions'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['estimated_stds'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['estimated_directions'].ndim == 1
            if space == 'sample':
                assert_consistency(data.d[space]['estimated_pairs'], '942eb6e6d84949042778819a5fcf5b03')
            if space == 'feature':
                assert_consistency(data.d[space]['estimated_pairs'], '7da8a6c533c3b97c689debdfb9d606e6')
        
class TestDataBlind(unittest.TestCase):
    """Test to verify that the inital data is created correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, (default = (50, 60))
        Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features).
    rank : int, (default = 2)
        Rank of the low-rank decomposition.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.rank = 2
        
    def test(self):
        mixed = np.random.normal(0, 1, size=self.shape)
        mixed[mixed < 0.5] = np.nan
        data = DataBlind(mixed=mixed, rank=self.rank)
        data.estimate() 
        for space in ['feature', 'sample']:
            assert data.d[space]['estimated_correlations'].shape == (data.d[space]['shape'][0], data.d[space]['shape'][0])
            assert data.d[space]['estimated_pairs'].shape[1] == 2
            assert data.d[space]['estimated_directions'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['estimated_stds'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['estimated_directions'].ndim == 1
            if space == 'sample':
                assert_consistency(data.d[space]['estimated_pairs'], '4519ddfb89b4dd6f377bd97395fe9b0b')
            if space == 'feature':
                assert_consistency(data.d[space]['estimated_pairs'], '1994f0d0dc435f66ec3bbc85473b2249')
        
    
if __name__ == '__main__':
    unittest.main()



    












