"""Bias matrix tests.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.bias import BiasLowRank, BiasUnconstrained


class TestBiasLowRank(unittest.TestCase):
    """Test to verify that all three models produce finite ndarrays.
    """
    def setUp(self):
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
        bias = BiasLowRank(self.shape,  self.rank, bias_model='image', image_source='trump.png').generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        
    def test_gaussian(self):
        bias = BiasLowRank(self.shape, self.rank, bias_model='gaussian', noise_amplitude=1.0).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        
    def test_bicluster(self):
        bias = BiasLowRank(self.shape, self.rank, bias_model='bicluster', n_clusters=(3,4)).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)


class TestBiasUnconstrained(unittest.TestCase):
    """Test to verify that both models produce finite ndarrays.
    """
    def setUp(self):
        self.shape = (50, 60)

    def _assert_shape(self, bias):
        assert bias['X'].shape == self.shape

    def _assert_finite(self, bias):
        assert np.isfinite(bias['X']).all() == True

    def _assert_ndarray(self, bias):
        assert type(bias['X']) == np.ndarray

    def _assert_dict(self, bias):
        assert type(bias) == dict
        
    def test_gaussian_finite(self):
        bias = BiasUnconstrained(self.shape, 'gaussian', noise_amplitude=1.0).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)
        
    def test_uniform_finite(self):
        bias = BiasUnconstrained(self.shape, 'uniform', fill_value=-1.5).generate()
        self._assert_dict(bias)
        self._assert_finite(bias)
        self._assert_ndarray(bias)
        self._assert_shape(bias)

    
if __name__ == '__main__':
    unittest.main()



