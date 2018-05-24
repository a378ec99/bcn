"""Missing values tests.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.missing import Missing

    
class TestMissing(unittest.TestCase):
    """Test to verify that all three models produce non-finite/finite ndarrays that are consistent.
    """
    def setUp(self):
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

    def test_NMAR(self):
        missing = Missing(self.shape, 'NMAR', p_censored=0.2).generate()
        self._assert_dict(missing)
        self._assert_not_finite(missing)
        self._assert_ndarray(missing)
        self._assert_shape(missing)

    def test_NoMissing(self):
        missing = Missing(self.shape, 'no-missing').generate()
        self._assert_dict(missing)
        self._assert_finite(missing)
        self._assert_ndarray(missing)
        self._assert_shape(missing)

    
if __name__ == '__main__':
    unittest.main()



