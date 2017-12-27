"""Bias testing.

Notes
-----
Defines two test classes that assert the functioning of the `bias` module.
"""
from __future__ import division, absolute_import


__all__ = ['TestBiasLowRank', 'TestBiasUnconstrained']

import unittest
import hashlib
#import sys
import numpy as np
#sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')
from bcn.bias import BiasLowRank, BiasUnconstrained


def _assert_consistency(X, true_md5):
    m = hashlib.md5()
    m.update(X)
    current_md5 = m.hexdigest()
    assert current_md5 == true_md5

        
class TestBiasLowRank(unittest.TestCase):
    """Test to verify that all three models produce finite ndarrays that are consistent.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (50, 60)
        Shape of the output bias matrix in the form of (n_samples, n_features).
    rank : int, default = 6
        Rank of the low-rank decomposition.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.rank = 6

    def _assert_shape(self, bias):
        assert bias['X'].shape == self.shape
        assert bias['usvt'][0].shape == (self.shape[0], self.rank)
        assert bias['usvt'][1].shape == (self.rank,)
        assert bias['usvt'][2].shape == (self.rank, self.shape[1])

    def _assert_finite(self, bias):
        assert np.isfinite(bias['X']).all() == True
        assert np.isfinite(bias['usvt'][0]).all() == True
        assert np.isfinite(bias['usvt'][1]).all() == True
        assert np.isfinite(bias['usvt'][2]).all() == True

    def _assert_ndarray(self, bias):
        assert type(bias['X']) == np.ndarray
        assert type(bias['usvt'][0]) == np.ndarray
        assert type(bias['usvt'][1]) == np.ndarray
        assert type(bias['usvt'][2]) == np.ndarray

    def _assert_dict(self, bias):
        assert type(bias) == dict

    def test_image(self):
        bias = BiasLowRank(self.shape,  self.rank, model='image', image_source='trump.png').generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        _assert_consistency(bias['X'], '1b321768958469659a3f65bd19f096cf')
        
    def test_gaussian(self):
        bias = BiasLowRank(self.shape, self.rank, model='gaussian', noise_amplitude=1.0).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        _assert_consistency(bias['X'], '64fd423a0625ae269d1b9e07f57eb31c')
        
    def test_bicluster(self):
        bias = BiasLowRank(self.shape, self.rank, model='bicluster', n_clusters=(3,4)).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        _assert_consistency(bias['X'], '3cc3ffcca30bda21e83043d0be8e03b9')


class TestBiasUnconstrained(unittest.TestCase):
    """Test to verify that both models produce finite ndarrays.

    Attributes
    ----------
    seed : int, optional (default = 42)
        Random seed of the whole test.
    shape : tuple of int, default = (50, 60)
        Shape of the output bias matrix in the form of (n_samples, n_features).
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)

    def _assert_shape(self, bias):
        assert bias['X'].shape == self.shape

    def _assert_finite(self, bias):
        assert np.isfinite(bias['X']).all() == True

    def _assert_ndarray(self, bias):
        assert type(bias['X']) == np.ndarray

    def _assert_dict(self, bias):
        assert type(bias) == dict

    def _assert_consistency(self, bias, true_md5):
        m = hashlib.md5()
        m.update(bias['X'])
        current_md5 = m.hexdigest()
        assert current_md5 == true_md5
        
    def test_gaussian_finite(self):
        bias = BiasUnconstrained(self.shape, 'gaussian', noise_amplitude=1.0).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        _assert_consistency(bias['X'], '274886b402bf99b7eef93466eb580f55')
        
    def test_uniform_finite(self):
        bias = BiasUnconstrained(self.shape, 'uniform', fill_value=-1.5).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        _assert_consistency(bias['X'], 'c550d003aa3c3eaedf8b0c8f6ffc6911')

    
if __name__ == '__main__':
    unittest.main()



