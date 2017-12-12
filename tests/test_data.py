"""Data testing.

Notes
-----
Defines a test class that assert the functioning of the `data` module.
"""
from __future__ import division, absolute_import


__all__ = ['Test_pair_subset', 'TestDataSimulated', 'TestDataBlind'] # , 'TestDataSimulatedLarge', 'TestDataBlindLarge', , 'TestStdEstimationImprovement'

import unittest
import hashlib
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')
from data import DataSimulated, DataBlind #, pair_subset # DataSimulatedLarge, DataBlindLarge, 


def _assert_consistency(X, true_md5):
    m = hashlib.md5()
    m.update(X)
    current_md5 = m.hexdigest()
    assert current_md5 == true_md5

'''    
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
        #print 'pairs', pairs
        #print 'subset_indices', subset_indices
        #print 'subset_pairs', subset_pairs
        #_assert_consistency(subset_pairs, '1')
'''
        
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

'''       
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
'''        
'''
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
'''
'''
class TestStdEstimationImprovement(unittest.TestCase):
    """Test to verify that when estimating standard deviations from DataSimulatedLarge there is an improvement.

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
        #np.random.seed(seed)
        self.rank = 2
        self.shape = (50, 60)

    def rmse(self, X, true):
        return np.sqrt(np.mean((X - true)**2))
    
    def test(self):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        for n, shape_factor in enumerate(range(1, 10)):
            #print n
            shape = tuple(np.asarray(self.shape) * shape_factor)
            data_large = DataSimulatedLarge(large_scale_shape=shape, rank=self.rank)
            data_large.estimate(true_pairs={'sample': data_large.d['sample']['true_pairs'], 'feature': data_large.d['feature']['true_pairs']}, true_directions={'sample': data_large.d['sample']['true_directions'], 'feature': data_large.d['feature']['true_directions']})
            data = DataBlind(mixed=data_large.d['sample']['mixed'], rank=self.rank)
            data.estimate(true_pairs={'sample': data_large.d['sample']['true_pairs'], 'feature': data_large.d['feature']['true_pairs']}, true_directions={'sample': data_large.d['sample']['true_directions'], 'feature': data_large.d['feature']['true_directions']})
            #print 'large estimate', data_large.d['sample']['estimated_stds'], data_large.d['sample']['estimated_stds'].shape
            #print 'normal estimate', data.d['sample']['estimated_stds'], data.d['sample']['estimated_stds'].shape
            #print 'true', data_large.d['sample']['true_stds'], data_large.d['sample']['true_stds'].shape
            large_rmsq = self.rmse(data_large.d['sample']['estimated_stds'], data_large.d['sample']['true_stds'])
            small_rmsq = self.rmse(data.d['sample']['estimated_stds'], data_large.d['sample']['true_stds'])
            ax.plot(n, large_rmsq, 'D', color='red')
            ax.plot(n, large_rmsq, '.', color='blue')
        fig.savefig('rmse_estimation_improvement_test_10')
'''        
    
if __name__ == '__main__':
    unittest.main()



    












