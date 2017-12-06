"""Missing values testing module.

Notes
-----
This module defines a test class that assert the functioning of the `missing` module.
"""
from __future__ import division, absolute_import


__all__ = ['TestMissing']

import unittest
import hashlib
import sys
import numpy as np
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')
from missing import Missing


def _assert_consistency(X, true_md5):
    m = hashlib.md5()
    m.update(X)
    current_md5 = m.hexdigest()
    assert current_md5 == true_md5

    
class TestMissing(unittest.TestCase):
    """Test to verify that all three models produce non-finite/finite ndarrays that are consistent.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (50, 60)
        Shape of the output missing values matrix in the form of (n_samples, n_features).
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        
    def _assert_shape(self, missing):
        assert missing['X'].shape == self.shape

    def _assert_finite(self, missing):
        assert np.isfinite(missing['X']).all() == True

    def _assert_not_finite(self, missing):
        assert np.isfinite(missing['X']).all() == False

    def _assert_ndarray(self, missing):
        assert type(missing['X']) == np.ndarray

    def _assert_dict(self, missing):
        assert type(missing) == dict

    def test_MAR(self):
        missing = Missing(self.shape, 'MAR', p_random=0.1).generate()
        self._assert_dict(missing)
        self._assert_not_finite(missing)
        self._assert_ndarray(missing)
        self._assert_shape(missing)
        _assert_consistency(missing['X'], '887918fd77dbe00d302b1105da156eb5')

    def test_NMAR(self):
        missing = Missing(self.shape, 'NMAR', p_censored=0.2).generate()
        self._assert_dict(missing)
        self._assert_not_finite(missing)
        self._assert_ndarray(missing)
        self._assert_shape(missing)
        _assert_consistency(missing['X'], '5a506bce5f99444f6ee3bed2e7e2b6cc')

    def test_NoMissing(self):
        missing = Missing(self.shape, 'no-missing').generate()
        self._assert_dict(missing)
        self._assert_finite(missing)
        self._assert_ndarray(missing)
        self._assert_shape(missing)
        _assert_consistency(missing['X'], 'c30c3f99eebe77a50cbd1ef6ef15a34b')

    def test_SCAN(self):
        pass

    
if __name__ == '__main__':
    unittest.main()



