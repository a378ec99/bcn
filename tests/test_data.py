"""Data testing module.

Notes
-----
This module defines a test class that assert the functioning of the `data` module.
"""
from __future__ import division, absolute_import


__all__ = ['Test_pair_subset', 'TestDataSimulated', 'TestDataBlind', 'TestDataSimulatedLarge', 'TestDataBlindLarge']

import unittest
import hashlib
import sys
import numpy as np
sys.path.append('/home/sohse/projects/PUBLICATION/GIT2/bcn')
from data import DataSimulated, DataSimulatedLarge, DataBlind, DataBlindLarge, pair_subset


def _assert_consistency(X, true_md5):
    m = hashlib.md5()
    m.update(X)
    current_md5 = m.hexdigest()
    assert current_md5 == true_md5

    
class Test_pair_subset(unittest.TestCase):
    """Test to verify that subset of pairs is selected appropriately.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)

    def test(self):
        pairs = np.asarray([[1, 2], [3, 4], [9, 1], [10, 11], [98, 99]])
        subset_indices = np.asarray([1, 2, 4, 9])
        subset_pairs = pair_subset(pairs, subset_indices, mode='subset_pairs')
        print 'pairs', pairs
        print 'subset_indices', subset_indices
        print 'subset_pairs', subset_pairs
        #_assert_consistency(subset_pairs, '1')

        
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
            #_assert_consistency(data.d[space]['estimated_pairs'], '1')
        data.guess()
        
        
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
            #_assert_consistency(data.d[space]['estimated_pairs'], '1')
        data.guess()

       
class TestDataSimulatedLarge(unittest.TestCase):
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
        self.shape = (4 * 50, 4 * 60)
        self.rank = 2

    def test(self):
        data = DataSimulatedLarge(large_scale_shape=self.shape, rank=self.rank)
        data.estimate()
        for space in ['feature', 'sample']:
            assert data.d[space]['estimated_correlations'].shape == (data.d[space]['shape'][0], data.d[space]['shape'][0])
            assert data.d[space]['estimated_pairs'].shape[1] == 2
            assert data.d[space]['estimated_directions'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['true_directions'].shape[0] == data.d[space]['true_pairs'].shape[0]
            assert data.d[space]['estimated_stds'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['true_stds'].shape[0] == data.d[space]['true_pairs'].shape[0]
            assert data.d[space]['estimated_directions'].ndim == 1
            #_assert_consistency(data.d[space]['estimated_pairs'], '1')
        data.guess()
        

class TestDataBlindLarge(unittest.TestCase):
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
        data = DataBlindLarge(large_scale_mixed=mixed, rank=self.rank)
        data.estimate()
        for space in ['feature', 'sample']:
            assert data.d[space]['estimated_correlations'].shape == (data.d[space]['shape'][0], data.d[space]['shape'][0])
            assert data.d[space]['estimated_pairs'].shape[1] == 2
            assert data.d[space]['estimated_directions'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['estimated_stds'].shape[0] == data.d[space]['estimated_pairs'].shape[0]
            assert data.d[space]['estimated_directions'].ndim == 1
            #_assert_consistency(data.d[space]['estimated_pairs'], '1')
        data.guess()

        
if __name__ == '__main__':
    unittest.main()



    












